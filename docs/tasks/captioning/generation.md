# Comic Panel Captioning Script

This script processes comic panel images to generate captions and descriptive lists using various vision-language models from Hugging Face.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Input Data Format](#input-data-format)
- [Output Format](#output-format)
- [Supported Models](#supported-models)
- [Troubleshooting](#troubleshooting)

## Overview

The script provides automated captioning for comic panels using state-of-the-art vision-language models. It can:
- Process full comic pages and automatically crop panels
- Generate detailed captions for each panel
- Extract lists of relevant items/details
- Handle batch processing with customizable splits
- Save results in both raw and processed formats

## Prerequisites

### Hardware Requirements
- CUDA-capable GPU (required)
- RAM requirements depend on chosen model and batch size
- Actual VRAM usage may vary based on implementation and settings

**Note**: Memory requirements can vary significantly based on:
- Model quantization settings
- Batch size configuration
- Input image resolution
- System configuration and available optimizations

### Software Requirements
- Python 3.8+
- CUDA Toolkit 11.8 or higher
- Git LFS (for model downloads)

### Required Python Packages
```bash
# Core dependencies
pip install torch torchvision 
pip install transformers pillow pandas numpy tqdm

# Model-specific dependencies
pip install flash-attention-2 bitsandbytes accelerate
```

To check installed packages:
```bash
pip list | grep -E "torch|transformers|pillow|pandas|numpy|tqdm|flash-attention|bitsandbytes|accelerate"
```

### Data structure
```
data/
├── predicts.caps/
│   └── [model-name]-cap/               # Here will be added the model predictions
└── datasets.unify/
    └── [subdb]/                        # Comic subdatabases
        └── [comic_no]/                 # Individual books
            └── [page_no].jpg           # Full comic pages
```

### Directory structure
```
benchmarks/
└── captioning/
    ├── prompts.py                      # Model prompts
    ├── generate_captions.py            # Main script
    └── postprocessing.py               # Postprocessing script

```

## Usage

### Basic Command
```bash
python benchmarks/captioning/generate_captions.py --model MODEL_NAME --num_splits N --index I [options]
```

### Required Arguments
- `--model`: Model choice ("minicpm2.6", "qwen2", "florence2", "idefics2", "idefics3")
- `--num_splits`: Total number of dataset splits
- `--index`: Index of current split (0-based)

### Optional Arguments
- `--override`: Override existing processed files
- `--batch_size`: Batch size for processing (default varies by model)
- `--num_workers`: Number of DataLoader workers (default: 4)
- `--save_txt`: Save raw results as text files
- `--save_csv`: Extract and save results to CSV files

### Example Commands
```bash
# Process 1/4 of dataset with MiniCPM (medium VRAM usage)
python benchmarks/captioning/generate_captions.py --model minicpm2.6 --num_splits 4 --index 0 --batch_size 1 --save_txt --save_csv

# Process with Qwen2VL (requires ~40GB VRAM)
python benchmarks/captioning/generate_captions.py --model qwen2 --num_splits 4 --index 1 --batch_size 2 --save_txt --save_csv

# Process with Florence2 (smallest VRAM usage)
python benchmarks/captioning/generate_captions.py --model florence2 --num_splits 4 --index 2 --batch_size 16 --save_txt --save_csv
```

## Input Data Format

### Panel Annotations CSV
Required columns in `compiled_panels_annotations.csv`:
```csv
subdb,comic_no,page_no,panel_no,x1,y1,x2,y2
subdb1,book1,1,1,100,100,300,300
```
- `subdb`: Subdatabase identifier
- `comic_no`: Book identifier
- `page_no`: Page number (will be zero-padded to 3 digits)
- `panel_no`: Panel number within the page
- `x1,y1,x2,y2`: Panel bounding box coordinates

### Image Files
- Format: JPG
- Location: `data/datasets.unify/[subdb]/[comic_no]/[page_no].jpg`
- Content: Full comic pages (panels will be cropped automatically)

### Prompt Templates
File: `utils/prompt.py`
- Contains model-specific prompting templates
- Required templates:
  - `base_prompt`
  - `minicpm26_prompt`
  - `idefics2_prompt`

## Output Format

### Directory Structure
```
data/predicts.caps/[model-name]-cap/
├── results/                           # Raw results (if --save_txt)
│   └── subdb_book_page_panel.txt
├── N_I_caption.csv                    # Processed captions (if --save_csv)
└── N_I_list.csv                       # Processed lists (if --save_csv)
```

### CSV Output Format
```csv
subdb,comic_no,page_no,panel_no,caption/items
subdb1,book1,1,1,"A person walking down the street"
...
```

## Supported Models

1. **MiniCPM-V-2.6**
   - Model ID: "openbmb/MiniCPM-V-2_6"
   - Parameters: 8.1B
   - Default batch size: 16
   - Architecture: Vision-Language model with instruction tuning

2. **Qwen2VL**
   - Model ID: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4"
   - Parameters: 12.6B (Quantized from 72B)
   - Default batch size: 8
   - Architecture: Vision-Language model with instruction tuning

3. **Florence2**
   - Model ID: "microsoft/Florence-2-large-ft"
   - Parameters: 0.77B
   - Default batch size: 64
   - Architecture: Vision-Language model with contrastive learning

4. **Idefics2**
   - Model ID: "HuggingFaceM4/idefics2-8b"
   - Parameters: 8.4B
   - Default batch size: 16
   - Architecture: Vision-Language model with open-vocabulary detection

5. **Idefics3**
   - Model ID: "HuggingFaceM4/Idefics3-8B-Llama3"
   - Parameters: 8.46B
   - Default batch size: 16
   - Architecture: Vision-Language model based on Llama3

### Hardware Requirements
- CUDA-capable GPU (required)
- RAM requirements depend on chosen model and batch size
- Actual VRAM usage may vary based on implementation and settings

**Note**: Memory requirements can vary significantly based on:
- Model quantization settings
- Batch size configuration
- Input image resolution
- System configuration and available optimizations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Try a smaller model
   - Free up GPU memory
   - Use `nvidia-smi` to monitor GPU memory

2. **Missing Files**
   - Verify directory structure
   - Check file permissions
   - Ensure all required files exist

3. **Model Loading Errors**
   - Check internet connection
   - Verify HuggingFace authentication (`huggingface-cli login`)
   - Clear transformers cache if needed
   - Ensure Git LFS is installed

### Getting Help
- Check the error message for specific details
- Verify all prerequisites are met
- Ensure input data follows the required format