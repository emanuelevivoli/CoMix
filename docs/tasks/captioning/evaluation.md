# Captioning and object list evaluation

The `captioning.py` script evaluates both caption generation and object list detection:

## Setup
```bash
# Using the existing eval environment
conda activate eval

# Install additional requirements
pip install evaluate nltk transformers torch
```

## Features
1. **Caption Evaluation**:
   - Metrics: BLEU-1/2/3/4, ROUGE-1/2/L, METEOR
   - Handles missing entries and alignment between predictions and ground truth
   - Outputs formatted scores to terminal

2. **Object List Evaluation**:
   - BERT-based similarity matching
   - Precision, Recall, and F1 scores at different thresholds (0.5-0.999)
   - Panel-level statistics

## Usage
```bash
python comix/evaluators/captioning.py \
    -p PRED_FILE \     # Prediction file prefix [minicpm, etc.]
    --nlp_only \       # Evaluate only captions
    --arm_only         # Evaluate only object lists
```

**Note**:
- If `-p` is not provided, the script uses ground truth as both GT and predictions (debug mode, all metrics will be 1.0)
- If neither `--nlp_only` nor `--arm_only` is specified, both evaluations will be executed
- Use either `--nlp_only` or `--arm_only` to run specific evaluations

## Input Format
Expects CSV files in the following structure:
```
data/predicts.caps/
├── gt_captions.csv          # Ground truth captions
├── gt_lists.csv            # Ground truth object lists
├── {PRED_FILE}_captions.csv # Model caption predictions
└── {PRED_FILE}_lists.csv   # Model object list predictions
```

Required columns:
- `comic_no`, `page_no`, `panel_no`, `subdb`: Panel identifiers
- `caption`: Panel caption text (for caption evaluation)
- `items`/`objects`: Comma-separated object lists (for object detection)

## Output
- Terminal display of evaluation metrics
- `missing_instances_captions.txt`: Log of unmatched panels in caption evaluation
- `missing_instances_lists.txt`: Log of unmatched panels in object list evaluation
