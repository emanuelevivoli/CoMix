# Comic Panel Captioning

## Overview
This documentation covers the captioning pipeline for comic panels, which includes:
1. Panel detection and cropping
2. Caption generation using Vision-Language Models (VLMs)
3. Post-processing of generated captions
4. Evaluation of results

## Pipeline Components

### 1. Data Preparation
- Input: Full comic pages from unified dataset structure
- Output: Cropped panels based on annotation coordinates
- Reference: Panel coordinates from `compiled_panels_annotations.csv`

### 2. Caption Generation
The generation process uses multiple VLMs for comparison:

#### Supported Models
- MiniCPM-V-2.6 (8.1B parameters)
- Qwen2VL (72B parameters, quantized)
- Florence2 (0.77B parameters)
- IDEFICS-2 (8B parameters)
- IDEFICS-3 (8.46B parameters)

For detailed model configurations and usage, see [generation.md](generation.md)

### 3. Post-Processing
The pipeline includes post-processing steps to:
- Clean and format generated captions
- Extract structured object lists
- Normalize output format

For implementation details, see [postprocessing.md](postprocessing.md)

### 4. Evaluation
Evaluation metrics include:
- Caption quality (BLEU, ROUGE, METEOR)
- Object detection accuracy
- Panel coverage statistics

For evaluation procedures and metrics, see [evaluation.md](evaluation.md)

## Directory Structure
```
data/
├── predicts.caps/
│   └── [model-name]-cap/
│       ├── results/
│       │   └── [subdb]_[book_no]_[page_no]_[panel_no].txt  # Raw results (if --save_txt), one for each panel
│       └── N_I_caption.csv                                 # Processed captions
│       └── N_I_list.csv                                    # Processed lists
└── datasets.unify/
    ├── compiled_panels_annotations.csv                   # Panel coordinates
    └── [subdb]/
        └── [comic_no]/
            └── [page_no].jpg
```


## Usage Instructions

### 1. Environment Setup
```bash
conda create -n caption python=3.8
conda activate caption
pip install -r requirements.txt
```

### 2. Running the Pipeline
```bash
# Generate captions
python benchmarks/captioning/generate_captions.py \
        --model MODEL_NAME \
        --num_splits N \
        --index I \
        [options]
# Post-process results
python benchmarks/captioning/postprocess.py \
        --input_dir data/predicts.caps/MODEL_NAME-cap \
        --output_dir data/predicts.caps/MODEL_NAME-cap-processed
# Evaluate results
python comix/evaluators/captioning.py \
        -p MODEL_NAME \
        [--nlp_only | --arm_only]
```


## Implementation Details

### Key Components
1. **Dataset Loading**: 
   - Reference: `DCMPanelDataset` class
   ```python:benchmarks/captioning/generate_captions.py
   startLine: 69
   endLine: 125
   ```

2. **Model Configuration**:
   - Reference: Model definitions and prompts
   ```python:benchmarks/captioning/generation.md
   startLine: 171
   endLine: 200
   ```

3. **Evaluation Metrics**:
   - Reference: Evaluation implementation
   ```python:comix/evaluators/captioning.py
   startLine: 90
   endLine: 125
   ```

## References
- [ComiCap: A VLMs pipeline for dense captioning of Comic Panels](https://arxiv.org/abs/2409.16159)
- Original implementation and benchmarks
- Related documentation in `/docs/tasks/captioning/`