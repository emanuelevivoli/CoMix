import os
import re
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from benchmarks.cap_val.utils.prompt import llama3_1_prompt_v2

class TextDataset(Dataset):
    """Dataset for processing text files from VLM outputs."""
    def __init__(self, annotations_df, text_directory):
        self.annotations = annotations_df
        self.text_directory = text_directory

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        subdb = row['subdb']
        book_no = row['comic_no']
        page_no = str(int(float(row['page_no']))).zfill(3)
        panel_no = row['panel_no']

        # Construct panel ID and text file path
        panel_id = f"{subdb}_{book_no}_{page_no}_{panel_no}"
        text_path = os.path.join(self.text_directory, f"{panel_id}.txt")

        try:
            with open(text_path, 'r') as file:
                text_content = file.read()
        except FileNotFoundError:
            print(f"Warning: File not found - {text_path}")
            text_content = ""

        return text_content, {
            'subdb': subdb,
            'book_no': book_no,
            'page_no': page_no,
            'panel_no': panel_no
        }

def extract_text(output):
    """
    Extract caption from the output string.
    Handles various markdown formats and cleans the output.
    """
    caption_patterns = [
        r'\*\*Caption of the Comic Panel\*\*\n(.*?)(?=\*\*|$)',
        r'\*\*Caption\*\*\n(.*?)(?=\*\*|$)'
    ]
    
    for pattern in caption_patterns:
        if match := re.search(pattern, output, re.DOTALL):
            content = match.group(1).strip()
            # Handle various markdown code block formats
            if backticks_match := re.search(r'```(?:caption|text)?\n?(.*?)\n?```', content, re.DOTALL):
                return backticks_match.group(1).strip()
            return content
    return ""

def extract_objects(output):
    """
    Extract objects and their attributes from the output string.
    Filters out empty rows and cleans the output.
    """
    object_patterns = [
        r'\*\*Objects and Attributes with Synonyms\*\*\n(.*?)(?=\*\*|$)',
        r'\*\*Objects with Attributes and Synonyms\*\*\n(.*?)(?=\*\*|$)',
        r'\*\*Object List with Synonyms\*\*\n(.*?)(?=\*\*|$)'
    ]
    
    for pattern in object_patterns:
        if match := re.search(pattern, output, re.DOTALL):
            content = match.group(1).strip()
            if csv_match := re.search(r'```(?:csv|list)?\n?(.*?)\n?```', content, re.DOTALL):
                csv_content = csv_match.group(1).strip()
                # Split into rows and filter out empty ones
                rows = [
                    row.strip() 
                    for row in csv_content.split('\n')
                    if row.strip() and not row.strip().startswith('<object>') # Skip header
                ]
                # Process each row and filter out empty items
                objects = []
                for row in rows:
                    items = [item.strip() for item in row.split(';')]
                    filtered_items = [item for item in items if item]  # Remove empty items
                    if filtered_items:  # Only add row if it has non-empty items
                        objects.append(filtered_items)
                return objects
    return []

def initialize_files(output_dir, split_info, override=False):
    """Initialize output CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    mode = 'w' if override else 'a'
    
    files = {
        'objects': (f'{split_info}_objects.csv', ['subdb', 'book_no', 'page_no', 'panel_no', 'object_no', 'objects']),
        'captions': (f'{split_info}_captions.csv', ['subdb', 'book_no', 'page_no', 'panel_no', 'caption']),
        'full': (f'{split_info}_full.csv', ['subdb', 'book_no', 'page_no', 'panel_no', 'result'])
    }
    
    file_handlers = {}
    writers = {}
    
    for key, (filename, headers) in files.items():
        filepath = os.path.join(output_dir, filename)
        file_handlers[key] = open(filepath, mode, newline='')
        writers[key] = csv.writer(file_handlers[key])
        
        if mode == 'w' or os.stat(filepath).st_size == 0:
            writers[key].writerow(headers)
    
    return file_handlers, writers

def main():
    parser = argparse.ArgumentParser(description='Process VLM outputs with LLaMA 3.1')
    parser.add_argument('-n', '--num_splits', type=int, required=True)
    parser.add_argument('-i', '--index', type=int, required=True)
    parser.add_argument('-o', '--override', action='store_true')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # Paths setup
    dataset_root = '/home/evivoli/projects/CoMix/benchmarks/cap_val/data'
    text_directory = f'{dataset_root}/anns/minicpm2.6-cap/results'
    output_dir = f'{dataset_root}/anns/minicpm2.6-cap'

    # Load and split data
    compiled_df = pd.read_csv(os.path.join(dataset_root, 'compiled_panels_annotations.csv'))
    split_dfs = np.array_split(compiled_df, args.num_splits)
    selected_df = split_dfs[args.index].reset_index(drop=True)

    # Initialize dataset and dataloader
    dataset = TextDataset(selected_df, text_directory)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Initialize LLaMA model
    model_id = "unsloth/Llama-3.1-Nemotron-70B-Instruct-bnb-4bit"
    print("Loading LLaMA 3.1 model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    # Initialize output files
    split_info = f"{args.num_splits}_{args.index}"
    file_handlers, writers = initialize_files(output_dir, split_info, args.override)

    # Processing loop
    with torch.no_grad():
        for batch_texts, batch_info in tqdm(dataloader, desc="Processing"):
            # Prepare messages
            messages = [[
                {"role": "system", "content": llama3_1_prompt_v2},
                {"role": "user", "content": text}
            ] for text in batch_texts]

            # Generate responses
            inputs = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                padding=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=500,
                pad_token_id=tokenizer.eos_token_id
            )
            
            responses = tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )

            # Process responses
            for response, info in zip(responses, batch_info):
                caption = extract_text(response)
                objects = extract_objects(response)

                # Write results
                writers['captions'].writerow([
                    info['subdb'], info['book_no'], 
                    info['page_no'], info['panel_no'], caption
                ])

                for obj_idx, obj in enumerate(objects):
                    writers['objects'].writerow([
                        info['subdb'], info['book_no'], 
                        info['page_no'], info['panel_no'], 
                        obj_idx, ','.join(obj)
                    ])

                writers['full'].writerow([
                    info['subdb'], info['book_no'], 
                    info['page_no'], info['panel_no'], response
                ])

                # Flush files
                for file in file_handlers.values():
                    file.flush()

    # Close files
    for file in file_handlers.values():
        file.close()

    print(f'\nProcessing complete for split {args.index + 1}/{args.num_splits}')

if __name__ == "__main__":
    main()
