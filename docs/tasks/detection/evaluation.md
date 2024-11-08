# Detection evaluation

Two scripts are available for evaluating detection predictions (`predicts.coco`):

## 1. detection.py
Evaluates single model predictions with detailed metrics:
- **Classes**: panel, character, text, face, all, all (filtered)
- **Metrics**: AP.50, AP.50-95, AR-10, AR-100
- **Output**: Terminal table, CSV, and/or XLSX file

```bash
# Setup
conda create -n eval python=3.8
conda activate eval
pip install -e .
pip install openpyxl

# Usage
python comix/evaluators/detection.py \
    -n DATASET_NAME \  # [eBDtheque, DCM, comics, popmanga]
    -s SPLIT \         # [val, test]
    -wn WEIGHTS_NAME \ # Model weights identifier
    --save N \         # Number of visualization images to save
    --xlsx \           # Save to Excel
    --plotting \       # Print results to terminal
    --no_layout        # Output results in CSV format instead of table
```

## 2. detection_batch.py
Batch evaluates multiple models across all datasets:
- **Input Structure**:
  ```
  predicts.coco/
  ├── eBDtheque/
  │   ├── model1/
  │   │   └── val.json
  │   └── model2/
  │       └── val.json
  ├── DCM/
  └── ...
  ```
- **Features**:
  - Automatically discovers all models in prediction folders
  - Evaluates AP50 scores for each class (panel, face, character, text)
  - Generates comparison matrices of models vs datasets
  - Supports both table and CSV output formats

```bash
python comix/evaluators/detection_batch.py \
    -gt data/comix.coco \            # Ground truth folder
    -pd data/predicts.coco \         # Predictions folder
    -o out/detection_results.xlsx \  # Output file
    --xlsx \                         # Save to Excel
    --plotting \                     # Print results to terminal
    --no_layout                      # Output results in CSV format instead of table
```

Both scripts save results in Excel format for further analysis.
