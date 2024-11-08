import os
import json
from tqdm import tqdm
import numpy as np
from contextlib import contextmanager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import sys
import argparse
from tabulate import tabulate

@contextmanager
def suppress_stdout():
    """Suppress console output."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Dataset mapping
dataset_mapping = {
    'eBDtheque': 'ebd',
    'comics': 'c100', 
    'popmanga': 'pop',
    'DCM': 'dcm'
}

def get_stats(coco_eval):
    stats = {
        "APi": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        # "AP75": coco_eval.stats[2],
        # "APs": coco_eval.stats[3],
        # "APm": coco_eval.stats[4],
        # "APl": coco_eval.stats[5],
        "ARi-1": coco_eval.stats[6],
        "ARi-10": coco_eval.stats[7],
        "ARi-100": coco_eval.stats[8],
        # "ARs": coco_eval.stats[9],
        # "ARm": coco_eval.stats[10],
        # "ARl": coco_eval.stats[11],
    }
    return stats

# Reverse the mapping to get full names from abbreviations
reverse_dataset_mapping = {v: k for k, v in dataset_mapping.items()}

# Define datasets and classes
datasets_abbr = ['c100', 'ebd', 'dcm', 'pop']  # Abbreviated dataset names
classes = ['panel', 'face', 'character', 'text']

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Evaluate COCO predictions.')
    parser.add_argument('-gt', '--ground_truth', type=str, default='data/comix.coco', help='Path to ground truth COCO JSON folder')
    parser.add_argument('-pd', '--predictions', type=str, default='data/predicts.coco', help='Path to predictions COCO JSON folder')
    parser.add_argument('-o', '--output', type=str, default='out/CDF_detection_perclass.xlsx', help='Path to save the Excel results')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--xlsx', action='store_true', help='Save results to Excel file')
    parser.add_argument('--plotting', action='store_true', help='Print evaluation results')
    parser.add_argument('--no_layout', action='store_true', help='Print results in CSV format (only with --plotting)')

    args = parser.parse_args()
    # Create a debug print function
    def debug_print(*message, sep=' '):
        if args.debug:
            print(*message, sep=sep)    
    
    gt_folder = args.ground_truth
    pred_folder = args.predictions
    output_excel = args.output

    print(f"Ground Truth Folder: {gt_folder}")
    print(f"Predictions Folder: {pred_folder}")
    print(f"Output Excel File: {output_excel}")

    # Collect models
    models = set()
    print("\nCollecting models from prediction folders...")
    for dataset_abbr in datasets_abbr:
        dataset_name = reverse_dataset_mapping.get(dataset_abbr, dataset_abbr)
        pred_dataset_folder = os.path.join(pred_folder, dataset_name)
        debug_print(f"Checking prediction folder: {pred_dataset_folder}")
        if not os.path.exists(pred_dataset_folder):
            debug_print(f"  [Warning] Prediction folder not found: {pred_dataset_folder}")
            continue
        # List subdirectories as models
        subdirs = [d for d in os.listdir(pred_dataset_folder) if os.path.isdir(os.path.join(pred_dataset_folder, d))]
        if not subdirs:
            debug_print(f"  [Warning] No model subdirectories found in: {pred_dataset_folder}")
            continue
        for model_name in subdirs:
            models.add(model_name)
            debug_print(f"  [Found Model] {model_name}")
    
    models = sorted(models)
    if not models:
        debug_print("\n[Error] No models found in the prediction folders. Please check the prediction folder paths and ensure subdirectories for models are present.")
        sys.exit(1)
    
    print(f"\nTotal Models Found: {len(models)}")
    print(models)

    # Initialize results dictionary
    results = {class_name: pd.DataFrame(index=models, columns=datasets_abbr) for class_name in classes}
    
    # Main loop
    for dataset_abbr in tqdm(datasets_abbr, desc="Datasets", position=0):
        dataset_name = reverse_dataset_mapping.get(dataset_abbr, dataset_abbr)
        debug_print(f"\nProcessing dataset: {dataset_abbr} ({dataset_name})")
        gt_json_path = os.path.join(gt_folder, f'{dataset_abbr}-val.json')
        if not os.path.exists(gt_json_path):
            debug_print(f"  [Warning] Ground truth file not found: {gt_json_path}")
            continue
        debug_print(f"  [Loading Ground Truth] {gt_json_path}")
        try:
            with suppress_stdout():
                coco_gt = COCO(gt_json_path)
            debug_print(f"  [Debug] Ground truth annotations: {len(coco_gt.anns)}")
            debug_print(f"  [Debug] Ground truth images: {len(coco_gt.imgs)}")
        except Exception as e:
            debug_print(f"  [Error] Failed to load ground truth JSON: {e}")
            continue

        # Build mapping from class names to category IDs
        categories = coco_gt.loadCats(coco_gt.getCatIds())
        cat_name_to_id = {cat['name'].lower(): cat['id'] for cat in categories}  # Convert to lowercase for case-insensitive matching
        debug_print(f"  [Categories in Ground Truth]: {list(cat_name_to_id.keys())}")

        # Iterate over each model
        for model_name in tqdm(models, desc=f"Models ({dataset_abbr})", position=1, leave=False):
            pred_json_path = os.path.join(pred_folder, dataset_name, model_name, 'val.json')
            if not os.path.exists(pred_json_path):
                debug_print(f"  [Warning] Prediction file not found: {pred_json_path}")
                continue
            debug_print(f"  [Evaluating Model] {model_name} on dataset {dataset_abbr}")

            try:
                # Load the prediction JSON file
                with open(pred_json_path, 'r') as f:
                    pred_data = json.load(f)
                # Add debug print
                debug_print(f"    [Debug] Loaded predictions file: {pred_json_path}")
                debug_print(f"    [Debug] Number of predictions: {len(pred_data if isinstance(pred_data, list) else pred_data.get('annotations', []))}")
                
                if 'annotations' in pred_data:
                    detections = pred_data['annotations']
                elif isinstance(pred_data, list):
                    detections = pred_data
                else:
                    debug_print(f"    [Error] Prediction JSON format not recognized for model {model_name}")
                    continue
                
                # Add debug print before loading predictions
                debug_print(f"    [Debug] Attempting to load {len(detections)} detections into COCO format")
                with suppress_stdout():
                    coco_pred = coco_gt.loadRes(detections)
                debug_print(f"    [Debug] Successfully loaded predictions: {len(coco_pred.anns)} annotations")
            except Exception as e:
                debug_print(f"    [Error] Failed to load prediction JSON for model {model_name}: {e}")
                continue

            with suppress_stdout():
                cocoEval = COCOeval(coco_gt, coco_pred, 'bbox')
                cocoEval.evaluate()
                cocoEval.accumulate()

            # Iterate over each class
            for class_name in tqdm(classes, desc=f"Classes ({model_name})", position=2, leave=False):
                class_name_lower = class_name.lower()
                if class_name_lower not in cat_name_to_id:
                    debug_print(f"    [Info] Class '{class_name}' not found in ground truth for dataset '{dataset_abbr}'. Skipping.")
                    results[class_name].loc[model_name, dataset_abbr] = np.nan
                    continue
                catId = cat_name_to_id[class_name_lower]
                
                cocoEval.params.catIds = [catId]

                # Add debug print
                debug_print(f"    [Debug] Evaluating class '{class_name}' (catId: {catId})")
                debug_print(f"    [Debug] Ground truth annotations for this class: {len([ann for ann in coco_gt.anns.values() if ann['category_id'] == catId])}")
                debug_print(f"    [Debug] Predictions for this class: {len([ann for ann in coco_pred.anns.values() if ann['category_id'] == catId])}")

                # Inside the class evaluation loop, replace the evaluation section with:
                try:
                    with suppress_stdout():
                        cocoEval.evaluate()
                        cocoEval.accumulate()
                        cocoEval.summarize()

                    # cat_name = coco_gt.loadCats(catId)[0]['name']
                    # num_images = len(coco_gt.getImgIds(catIds=[catId]))
                    # num_instances = len(coco_gt.getAnnIds(catIds=[catId]))

                    stats = get_stats(cocoEval)

                    debug_print(f"    [Warning] No predictions for class '{class_name}'")
                    results[class_name].loc[model_name, dataset_abbr] = stats['AP50']

                except Exception as e:
                    debug_print(f"    [Error] Evaluation failed: {e}")
                    results[class_name].loc[model_name, dataset_abbr] = np.nan
    

    # Process and display/save results
    for class_name in classes:
        df = results[class_name]
        # Format the mAP values to display with 3 decimal places
        df_formatted = df.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
        
        # Print results if plotting is enabled
        if args.plotting:
            print(f"\nResults for class: {class_name}")
            if args.no_layout:
                print(df_formatted.to_csv())
            else:
                print(tabulate(df_formatted, headers='keys', tablefmt='pipe', showindex=True))
            print("\n")

    # Save to Excel if enabled
    if args.xlsx:
        try:
            print("\nSaving results to Excel...")
            with pd.ExcelWriter(args.output) as writer:
                for class_name in classes:
                    df = results[class_name]
                    df_formatted = df.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
                    df_formatted.to_excel(writer, sheet_name=class_name)
                    print(f"  [Saved] Sheet for class '{class_name}'")
            print(f"\n[Success] Results saved to '{args.output}'")
        except Exception as e:
            print(f"\n[Error] Failed to save Excel file: {e}")

if __name__ == "__main__":
    main()
