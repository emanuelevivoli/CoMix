#!/usr/bin/env python3
# captioning.py

"""
Unified Captioning Script for MiniCPM-V-2.6, Qwen2VL, Florence2, Idefics2, and Idefics3 Models

This script processes comic panel images to generate captions and lists using either
the MiniCPM-V-2.6, Qwen2VL, Florence2, Idefics2, or Idefics3 model from Hugging Face.
The model can be selected via command-line arguments.

Usage:
    python captioning.py --model "minicpm2.6" --num_splits N --index I [additional args]

Arguments:
    --model: Specify the model to use. Choices are "minicpm2.6", "qwen2", "florence2", "idefics2", or "idefics3".
    --num_splits: Total number of splits for the dataset.
    --index: Index of the current split (0-based).

Additional arguments may include:
    --save_txt: Save raw results as txt files.
    --save_csv: Extract and save results to CSV files.
    --override: Override existing processed files.
    --batch_size: Batch size for processing. (Default: 1 for minicpm, 2 for qwen2, 16 for florence2, etc.)
    --num_workers: Number of workers for DataLoader.

Example:
    python captioning.py --model "minicpm2.6" --num_splits 4 --index 0 --batch_size 16 --save_txt --override
    python captioning.py --model "qwen2" --num_splits 4 --index 1 --override --batch_size 2 --save_txt --save_csv
    python captioning.py --model "florence2" --num_splits 4 --index 2 --override --batch_size 16 --save_txt
    python captioning.py --model "idefics2" --num_splits 4 --index 3 --override --batch_size 16 --save_csv
    python captioning.py --model "idefics3" --num_splits 4 --index 4 --override --batch_size 16 --save_txt --save_csv
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
import torch
import re
import csv
import tempfile
import numpy as np

# Model-specific imports
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
)
from transformers import Qwen2VLForConditionalGeneration
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
from transformers import AutoModelForVision2Seq
from transformers.image_utils import load_image
from qwen_vl_utils import process_vision_info  # Ensure this is correctly installed

from prompts import (
    base_prompt,
    minicpm26_prompt,
    idefics2_prompt
)

# === Helper Classes and Functions ===

class DCMPanelDataset(Dataset):
    """Dataset for Comic Panels based on compiled CSV annotations."""
    def __init__(self, root, annotations_df, transform=None, config=None):
        """
        Args:
            root (string): Root directory where images are stored.
            annotations_df (DataFrame): Subset of the compiled CSV DataFrame.
            transform (callable, optional): Optional transform to be applied on a panel.
            config (dict, optional): Configuration dictionary for dataset processing.
        """
        self.root = root
        self.transform = transform
        self.annotations = annotations_df

        if config and not config.get('override', False):
            self.annotations = self.removed_done_panels(self.annotations, config)
    
    def removed_done_panels(self, panels, config):
        """Remove panels that have already been processed based on existing CSV files."""
        caption_csv = config.get('caption_csv')
        list_csv = config.get('list_csv')

        processed = set()

        for file_name in [caption_csv, list_csv]:
            if os.path.exists(file_name):
                df = pd.read_csv(file_name)
                key_tuples = zip(df['subdb'], df['comic_no'], df['page_no'], df['panel_no'])
                processed.update(key_tuples)
        
        # Create a tuple for each row in the panels DataFrame
        panels['key'] = list(zip(panels['subdb'], panels['comic_no'], panels['page_no'], panels['panel_no']))

        # Filter out rows where the key is in the processed set
        original_count = len(panels)
        panels = panels[~panels['key'].isin(processed)].copy()

        # Drop the auxiliary key column as it's no longer needed
        panels.drop(columns=['key'], inplace=True)

        print(f'Removed {original_count - len(panels)} already processed panels.')
        return panels

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.annotations.iloc[idx]
        subdb = row['subdb']
        comic_no = row['comic_no']
        page_no = str(int(float(row['page_no']))).zfill(3)  # Handle if page_no is float
        panel_no = row['panel_no']
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        
        # Construct the image path
        image_path = os.path.join(self.root, subdb, comic_no, f"{page_no}.jpg")
        if not os.path.exists(image_path):
            # Provide an empty image if the image is not found
            return Image.new('RGB', (512, 512)), {
                'IMAGE_PATH': image_path,
                'subdb': subdb,
                'comic_no': comic_no,
                'page_no': page_no,
                'panel_no': panel_no,
                'bbox': (x1, y1, x2, y2)
            }
        
        # Open the image
        image = Image.open(image_path).convert('RGB')
        
        # Crop the panel using the bounding box
        panel = image.crop((x1, y1, x2, y2))
        
        if self.transform:
            panel = self.transform(panel)
        
        return panel, {
            'subdb': subdb,
            'comic_no': comic_no,
            'page_no': page_no,
            'panel_no': panel_no,
            'bbox': (x1, y1, x2, y2)
        }

def extract_caption(output):
    """Extract the caption text from the given output string."""
    # Try to extract from a code block starting with ```caption
    caption_match = re.search(r'```caption\s+(.*?)\s+```', output, re.DOTALL)
    if caption_match:
        return caption_match.group(1).strip()
    else:
        # Extract text starting with 'Caption:' up to '---', '```list', or end of string
        caption_match = re.search(r'Caption:\s*(.*?)(?=\n---|\n```list|$)', output, re.DOTALL)
        if caption_match:
            return caption_match.group(1).strip()
    return None

def extract_list(output):
    """Extract the list of strings from the given output string."""
    list_match = re.search(r'```list\s+(.*?)\s+```', output, re.DOTALL)
    if list_match:
        return [item.strip() for item in list_match.group(1).split('\n') if item.strip()]
    return []

def extract_assistant(output):
    """
    Extract the assistant response from the given output string.
    It starts with the `Assistant:` and ends at the end of the string.
    """
    assistant_match = re.search(r'Assistant:\s+(.*)', output, re.DOTALL)
    if assistant_match:
        return assistant_match.group(1).strip()
    return None

class DotDict(dict):
    """Dictionary with attribute-style access."""
    def __getattr__(self, name):
        return self[name]
    
    def __setattr__(self, name, value):
        self[name] = value

# === Model Strategy Classes ===

class CaptioningModel:
    """Base class for captioning models."""
    def initialize(self):
        """Initialize the model. To be implemented by subclasses."""
        raise NotImplementedError
    
    def infer(self, images, prompt):
        """Generate captions and lists for given images. To be implemented by subclasses."""
        raise NotImplementedError

    def postprocess(self, results):
        """Postprocess the raw results to extract captions and lists."""
        captions = [extract_caption(content) for content in results]
        items_list = [extract_list(content) for content in results]
        return captions, items_list

    @staticmethod
    def check_broken(text):
        """
        Check if the generated text is broken or uninformative.
        Text should not be None or empty.
        """
        if not text: return True
        # minicpm
        if "i'm sorry" in text.lower(): return True
        if "upload the image" in text.lower(): return True
        if "please provide" in text.lower(): return True
        if "provide the image" in text.lower(): return True
        # paligemma
        if "unanswerable" in text.lower(): return True
        # florence 2
        if "completely black" in text.lower(): return True
        return False

class MiniCPMModel(CaptioningModel):
    """Captioning using MiniCPM-V-2.6 model."""
    def __init__(self, model_id="openbmb/MiniCPM-V-2_6", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.model = None
    
    def initialize(self):
        print("Loading the MiniCPM-V-2.6 model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model = self.model.eval()
        self.model.to(self.device)
        print("MiniCPM-V-2.6 model and tokenizer loaded successfully.")
    
    def infer(self, pil_imgs, prompt):
        """Generate caption and list for a batch of images."""
        system_prompt = "Answer in detail."
        batch_content = [[
            system_prompt,
            pil_img,
            prompt
        ] for pil_img in pil_imgs]
            
        msgs = [[{'role': 'user', 'content': content}] for content in batch_content]
    
        try:
            results = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,  # Enable sampling for diverse outputs
                stream=False     # Set to True if you want to handle streaming
            )
        except Exception as e:
            print(f"Error during MiniCPM inference: {e}")
            return ["" for _ in pil_imgs]  # Return empty results for the batch
        
        return results

# Define other model classes similarly (Qwen2VLModel, Florence2Model, Idefics2Model, Idefics3Model)
# Ensure each class implements initialize, infer, and inherits postprocess from CaptioningModel

class Qwen2VLModel(CaptioningModel):
    """Captioning using Qwen2VL model."""
    def __init__(self, model_id="Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
    
    def initialize(self):
        print("Loading the Qwen2VL model and processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.model = self.model.eval()
        self.model.to(self.device)
        print("Qwen2VL model and processor loaded successfully.")
    
    def infer(self, image_paths, prompt):
        """Generate captions and lists for a batch of images."""
        messages = []
        for image_url in image_paths:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_url  # Pass the image file URL
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ])
        
        input_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        
        # Process the vision information
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = self.processor(
            text=input_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Inference: Generation of the output
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        
        # Trim the generated tokens to exclude the input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the outputs
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        return output_texts

class Florence2Model(CaptioningModel):
    """Captioning using Florence2 model."""
    def __init__(self, model_id="microsoft/Florence-2-large-ft", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
        self.prompt = '<MORE_DETAILED_CAPTION>'
    
    def initialize(self):
        print("Loading the Florence2 model and processor...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        self.model = self.model.eval()
        self.model.to(self.device)
        print("Florence2 model and processor loaded successfully.")
    
    def infer(self, images, prompt=None):
        """Generate captions for a batch of images."""
        if prompt is None:
            prompt = self.prompt
        
        results = []
        
        for img in images:
            inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            res_dict = self.processor.post_process_generation(
                generated_text, 
                task=prompt, 
                image_size=(img.width, img.height)
            )
            res = res_dict.get('<MORE_DETAILED_CAPTION>', '')

            results.append(res)

        return results
    

class Idefics2Model(CaptioningModel):
    """Captioning using Idefics2 model."""
    def __init__(self, model_id="HuggingFaceM4/idefics2-8b", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
    
    def initialize(self):
        print("Loading the Idefics2 model and processor...")
        # BitsAndBytesConfig for quantization
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = Idefics2ForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
        self.processor = Idefics2Processor.from_pretrained(self.model_id)
        self.model = self.model.eval()
        self.model.to(self.device)
        print("Idefics2 model and processor loaded successfully.")
    
    def infer(self, batch_imgs, prompt):
        """Generate captions for a batch of images."""
        results = []
        
        for img in batch_imgs:
            # Prepare messages (assuming similar to previous models)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1000,
                do_sample=False,
                num_beams=3
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            res = extract_assistant(generated_text)

            results.append(res)

        return results


class Idefics3Model(CaptioningModel):
    """Captioning using Idefics3 model."""
    def __init__(self, model_id="HuggingFaceM4/Idefics3-8B-Llama3", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
    
    def initialize(self):
        print("Loading the Idefics3 model and processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            _attn_implementation="flash_attention_2",
        )
        self.model = self.model.eval()
        self.model.to(self.device)
        print("Idefics3 model and processor loaded successfully.")
    
    def infer(self, images, prompt):
        """Generate captions for a batch of images."""
        captions = []
        items_list = []  # Idefics3 may or may not generate lists depending on implementation

        for img in images:
            msg = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }]
        
            msg = self.processor.apply_chat_template(msg, add_generation_prompt=True)

            # Prepare inputs
            inputs = self.processor(
                text=msg,
                images=[img],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        
            # Inference
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        
            # Decode
            batch_content = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )

            results = [extract_assistant(content) for content in batch_content]

            return results


# === Main Workflow ===

def arg_parser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Split dataset and run on specified GPU with selected model.")
    parser.add_argument('--model', type=str, required=True, 
                       choices=["minicpm2.6", "qwen2", "florence2", "idefics2", "idefics3"], 
                       help='Model to use: "minicpm2.6", "qwen2", "florence2", "idefics2", or "idefics3".')
    parser.add_argument('-n', '--num_splits', type=int, required=True, 
                       help='Total number of splits.')
    parser.add_argument('-i', '--index', type=int, required=True, 
                       help='Index of the current split (0-based).')
    parser.add_argument('-o', '--override', action='store_true', 
                       help='Override existing processed files.')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for processing. If not specified, uses model-specific defaults.')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of workers for DataLoader.')
    parser.add_argument('--save_txt', action='store_true', 
                       help='Save raw results as txt files.')
    parser.add_argument('--save_csv', action='store_true', 
                       help='Extract and save results to CSV files.')
    args = parser.parse_args()
    
    # Set default batch sizes based on model if not specified
    if args.batch_size is None:
        model_batch_sizes = {
            "minicpm2.6": 16,
            "qwen2": 8,
            "florence2": 64,
            "idefics2": 16,
            "idefics3": 16
        }
        args.batch_size = model_batch_sizes[args.model]
    
    return args

def save_txt_results(results, batch_info, output_subdir, override):
    """Save raw results to txt files."""
    txt_dir = os.path.join(output_subdir, 'results')
    os.makedirs(txt_dir, exist_ok=True)
    
    for idx, info in enumerate(batch_info):
        txt_filename = f'{info["subdb"]}_{info["comic_no"]}_{info["page_no"]}_{info["panel_no"]}.txt'
        txt_path = os.path.join(txt_dir, txt_filename)
        
        if os.path.exists(txt_path) and not override:
            print(f"TXT file {txt_path} already exists. Skipping due to no override.")
            continue
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(results[idx])
    print(f"Saved raw results to txt files in {txt_dir}.")

def save_csv_results(captions, items_list, batch_info, writer_caption, writer_lists, override):
    """Save extracted captions and lists to CSV files."""
    for idx, info in enumerate(batch_info):
        caption = captions[idx] if captions[idx] else ""
        items = items_list[idx] if items_list[idx] else []
        
        # Write to CSV
        writer_caption.writerow([info['subdb'], info['comic_no'], info['page_no'], info['panel_no'], caption])
        writer_lists.writerow([info['subdb'], info['comic_no'], info['page_no'], info['panel_no'], ",".join(items)])
        
        # Visualization and Debugging
        print("\n\n\n")
        print(f"--- Subdataset: {info['subdb']} | Book: {info['comic_no']} | Page: {info['page_no']} | Panel: {info['panel_no']} ---")
        # Uncomment the following line if running in an environment that supports display
        # display(img)
        print("API Response:")
        print(f"Caption: {caption}")
        print("Extracted List:")
        print(items)

def initialize_csv_files(caption_csv, list_csv, save_csv):
    """Initialize CSV files with headers if necessary."""
    if save_csv:
        # Open CSV files for appending
        caption_file = open(caption_csv, mode='a', newline='', encoding='utf-8')
        list_file = open(list_csv, mode='a', newline='', encoding='utf-8')
        writer_caption = csv.writer(caption_file)
        writer_lists = csv.writer(list_file)
        # Write headers if files are empty
        if os.stat(caption_csv).st_size == 0:
            writer_caption.writerow(['subdb', 'comic_no', 'page_no', 'panel_no', 'caption'])
        if os.stat(list_csv).st_size == 0:
            writer_lists.writerow(['subdb', 'comic_no', 'page_no', 'panel_no', 'items'])
    else:
        caption_file = None
        list_file = None
        writer_caption = None
        writer_lists = None
    return caption_file, list_file, writer_caption, writer_lists

def close_csv_files(caption_file, list_file):
    """Close CSV files."""
    if caption_file:
        caption_file.close()
    if list_file:
        list_file.close()

def main():
    # Parse command-line arguments
    args = arg_parser()
    model_name = args.model.lower()
    n = args.num_splits
    i = args.index
    override = args.override
    batch_size = args.batch_size
    num_workers = args.num_workers
    save_txt = args.save_txt
    save_csv = args.save_csv

    # Validate split index
    if i < 0 or i >= n:
        raise ValueError(f"Split index i={i} is out of range for n={n} splits.")

    # Define the root path of your dataset
    dataset_root = '/home/evivoli/projects/CoMix/benchmarks/cap_val/data'  # Update this path as needed
    utils_root = '/home/evivoli/projects/CoMix/benchmarks/cap_val/utils'  # Update this path as needed
    compiled_csv_path = os.path.join(dataset_root, 'compiled_panels_annotations.csv')

    # Load the compiled CSV
    try:
        compiled_df = pd.read_csv(compiled_csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Display the first few rows
    print("First 5 entries of the compiled annotations:")
    print(compiled_df.head())

    # Display summary statistics
    print("Summary statistics:")
    print(compiled_df.describe())

    # Check for any missing values
    print("Missing values in each column:")
    print(compiled_df.isnull().sum())

    # Split the DataFrame into n parts
    split_dfs = np.array_split(compiled_df, n)

    # Select the i-th split
    selected_df = split_dfs[i].reset_index(drop=True)
    print(f"Processing split {i+1}/{n} with {len(selected_df)} entries.")

    # Define output directory and filenames with split info
    output_dir = os.path.join(dataset_root, 'anns')
    os.makedirs(output_dir, exist_ok=True)

    if model_name in ["minicpm2.6", "qwen2", "florence2", "idefics2", "idefics3"]:
        output_subdir = os.path.join(output_dir, f'{model_name}-cap')
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    os.makedirs(output_subdir, exist_ok=True)
    caption_csv = os.path.join(output_subdir, f'{n}_{i}_caption.csv')
    list_csv = os.path.join(output_subdir, f'{n}_{i}_list.csv')

    # Initialize the dataset with the selected split
    dataset = DCMPanelDataset(
        root=dataset_root,
        annotations_df=selected_df,
        transform=Resize((512, 512)),
        config=DotDict({
            'override': override,
            'caption_csv': caption_csv,
            'list_csv': list_csv,
        })
    )

    # Define collate function based on model
    if model_name in ["minicpm2.6", "qwen2", "florence2", "idefics2", "idefics3"]:

        def pil_collate(batch):
            """Custom collate function to handle batches of PIL Images and their info."""
            imgs, infos = zip(*batch)
            return list(imgs), list(infos)

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pil_collate
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Initialize CSV files with headers if needed
    caption_file, list_file, writer_caption, writer_lists = initialize_csv_files(caption_csv, list_csv, save_csv)

    # Retrieve the prompt from prompt.txt or model-specific prompt files
    
    if model_name == "minicpm2.6":
        prompt = minicpm26_prompt
    elif model_name in ["qwen2", "florence2","idefics3"]:
        prompt = base_prompt
    elif model_name == "idefics2":
        prompt = idefics2_prompt
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Initialize the Selected Model
    if model_name == "minicpm2.6":
        model = MiniCPMModel()
    elif model_name == "qwen2":
        model = Qwen2VLModel()
    elif model_name == "florence2":
        model = Florence2Model()
    elif model_name == "idefics2":
        model = Idefics2Model()
    elif model_name == "idefics3":
        model = Idefics3Model()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.initialize()

    # Open CSV files for appending is handled in initialize_csv_files
    # Create a temporary directory for storing cropped images (if needed)
    temp_dir = tempfile.TemporaryDirectory()

    # Processing loop
    with torch.no_grad():
        for batch_imgs, batch_info in tqdm(data_loader, desc=f"Processing Panels with {model_name}"):
            # Inference
            if model_name == "minicpm2.6":
                results = model.infer(batch_imgs, prompt)
            elif model_name == "qwen2":
                # Save each image to the temporary directory and prepare file URLs
                image_urls = []
                for img, info in zip(batch_imgs, batch_info):
                    temp_image_filename = f"temp_{info['subdb']}_{info['comic_no']}_{info['page_no']}_{info['panel_no']}.jpg"
                    temp_image_path = os.path.join(temp_dir.name, temp_image_filename)
                    img.save(temp_image_path, format='JPEG')
                    image_url = f"file://{temp_image_path}"
                    image_urls.append(image_url)
                results = model.infer(image_urls, prompt)
            else:
                results = model.infer(batch_imgs, prompt)
            
            # Save raw results to txt files if requested
            if save_txt:
                save_txt_results(results, batch_info, output_subdir, override)

            # Perform postprocess and save to CSV if requested
            if save_csv:
                captions, items_list = model.postprocess(results)
                save_csv_results(captions, items_list, batch_info, writer_caption, writer_lists, override)
            
            # Flush CSV files to ensure progress is not lost
            if save_csv:
                caption_file.flush()
                list_file.flush()

    # Close the CSV files
    close_csv_files(caption_file, list_file)

    # Clean up the temporary directory
    temp_dir.cleanup()

    if save_csv:
        print(f'\nFinished writing to {caption_csv}')
        print(f'Finished writing to {list_csv}')
    if save_txt:
        print(f'\nFinished writing txt files to {os.path.join(output_subdir, "results")}')
    if not (save_csv or save_txt):
        print("No output files were saved. Use --save_txt and/or --save_csv to save results.")

if __name__ == "__main__":
    main()