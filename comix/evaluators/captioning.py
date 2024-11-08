import torch
import pandas as pd
from tqdm import tqdm

from collections import defaultdict
from typing import List, Any, Tuple
from dataclasses import dataclass, field

import evaluate
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Define NLP metrics thresholds
THRESHOLD = 0.0

@dataclass
class Metric:
    """
    Base class for metrics to evaluate captions.
    """
    name: str = field(init=False)
    inpyt_type: str = field(init=False)
    values: List[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: float = 0.0
    average: float = 0.0

    def update(self, value: float, batch_size: int = 1) -> None:
        self.values.append(value)
        self.running_total += value * batch_size
        self.num_updates += batch_size
        self.average = self.running_total / self.num_updates

    def calculate_and_update(self, targets: Any, predictions: Any) -> float:
        raise NotImplementedError


# Define different NLP metrics classes with specific settings
class BLEUMetric(Metric):
    def __init__(self, max_order: int):
        super().__init__()
        self.name = f"bleu{max_order}" if max_order > 1 else "bleu"
        self.metric = evaluate.load("bleu")
        self.max_order = max_order
    
    def calculate_and_update(self, targets: List[str], predictions: List[str]) -> float:
        bleu = self.metric.compute(predictions=predictions, references=targets, max_order=self.max_order)["bleu"]
        self.update(bleu, len(targets))
        return bleu


class ROUGEMetric(Metric):
    def __init__(self, variant: str):
        super().__init__()
        self.name = f"rouge{variant}" if variant != "L" else "rougel"
        self.metric = evaluate.load("rouge")
        self.variant = f"rouge{variant}"
    
    def calculate_and_update(self, targets: List[str], predictions: List[str]) -> float:
        scores = self.metric.compute(predictions=predictions, references=targets)
        rouge_score = scores[self.variant]
        self.update(rouge_score, len(targets))
        return rouge_score


class METEORMetric(Metric):
    name: str = "meteor"

    def __init__(self):
        super().__init__()
        self.metric = evaluate.load("meteor")

    def calculate_and_update(self, targets: List[str], predictions: List[str]) -> float:
        meteor = self.metric.compute(predictions=predictions, references=targets)["meteor"]
        self.update(meteor, len(targets))
        return meteor

# Helper functions
def filter_caption(caption) -> str:
    """Filter and clean caption text."""
    # Handle NaN or empty captions
    if pd.isna(caption) or not isinstance(caption, str): return ""
    
    # Process valid caption string
    filtered_sentences = [s.strip() for s in caption.split('. ') if s.strip()]
    return '. '.join(filtered_sentences)

def load_data(gt_path: str, pred_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gt_data = pd.read_csv(gt_path)
    pred_data = pd.read_csv(pred_path)
    return gt_data, pred_data


def identify_missing_entries(gt_data: pd.DataFrame, pred_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing_vals_gt = []
    missing_vals_pred = []

    # Set to store all (comic_no, page_no, panel_no) combinations
    gt_combinations = set(tuple(row) for row in gt_data[["comic_no", "page_no", "panel_no"]].values)
    pred_combinations = set(tuple(row) for row in pred_data[["comic_no", "page_no", "panel_no"]].values)

    # Identify missing instances in ground truth and prediction
    for combo in gt_combinations - pred_combinations:
        missing_vals_pred.append({'comic_no': combo[0], 'page_no': combo[1], 'panel_no': combo[2]})
    for combo in pred_combinations - gt_combinations:
        missing_vals_gt.append({'comic_no': combo[0], 'page_no': combo[1], 'panel_no': combo[2]})

    return pd.DataFrame(missing_vals_gt), pd.DataFrame(missing_vals_pred)


def save_missing_instances(missing_gt: pd.DataFrame, missing_pred: pd.DataFrame, type: str = "captions"):
    with open(f"missing_instances_{type}.txt", "w") as file:
        if not missing_gt.empty:
            file.write(f"Missing in {type} Ground Truth:\n")
            for _, row in missing_gt.iterrows():
                file.write(f"Comic: {row['comic_no']}, Page: {row['page_no']}, Panel: {row['panel_no']}\n")
            file.write("\n")
        
        if not missing_pred.empty:
            file.write(f"Missing in {type} Predictions:\n")
            for _, row in missing_pred.iterrows():
                file.write(f"Comic: {row['comic_no']}, Page: {row['page_no']}, Panel: {row['panel_no']}\n")
    print("Missing instances saved to missing_instances.txt")


def print_scores(scores_dict):
    """Print evaluation scores in a clean, formatted way."""
    print("\n=== Caption Evaluation Scores ===\n")
    
    # First print individual scores
    print("Individual Metric Scores:")
    print("-" * 40)
    metrics_order = [
        ('BLEU-1', 'bleu1'), 
        ('BLEU-2', 'bleu2'),
        ('BLEU-3', 'bleu3'),
        ('BLEU-4', 'bleu4'),
        ('ROUGE-L', 'rougel'),
        ('ROUGE-1', 'rouge1'),
        ('ROUGE-2', 'rouge2'),
        ('METEOR', 'meteor')
    ]
    
    for display_name, metric_key in metrics_order:
        if metric_key in scores_dict:
            score = scores_dict[metric_key][0]
            # Convert numpy float to regular float if necessary
            score = float(score) if hasattr(score, 'item') else score
            print(f"{display_name:<10} : {score:.4f}")
    
    print("\n")


def evaluate_captions(gt_data: pd.DataFrame, pred_data: pd.DataFrame, metrics_names: List[str]) -> None:
    gts = []
    
    # Create merged dataframe to ensure alignment
    merged_df = pd.merge(
        gt_data,
        pred_data,
        on=['comic_no', 'page_no', 'panel_no', 'subdb'],
        suffixes=('_gt', '_pred'),
        how='inner'
    )
    
    hyps, refs = [], []
    for _, row in merged_df.iterrows():
        gt_caption = row['caption_gt'] 
        pred_caption = row['caption_pred']
        
        # Skip if either caption is empty/NaN
        if pd.isna(gt_caption) or pd.isna(pred_caption):
            continue
            
        gts.append({
            'gpt_caption': filter_caption(gt_caption),
            'pred_caption': filter_caption(pred_caption)
        })
        hyps.append(pred_caption)
        refs.append(gt_caption)

    # Continue with evaluation only if we have valid pairs
    if not gts:
        print("Warning: No valid caption pairs found for evaluation")
        return

    metrics = build_metrics(metrics_names)
    scores = defaultdict(list)
    for metric in metrics:
        val = metric.calculate_and_update(refs, hyps)
        scores[metric.name].append(val)
    
    # Print scores using the new formatting function
    print_scores(scores)


def build_metrics(metrics_names: List[str]) -> List[Metric]:
    metrics = []
    for metric_name in metrics_names:
        if metric_name.startswith("bleu"):
            order = int(metric_name.replace("bleu", "")) if metric_name != "bleu" else 1
            metrics.append(BLEUMetric(order))
        elif metric_name.startswith("rouge"):
            variant = metric_name.replace("rouge", "").upper() if metric_name != "rougel" else "L"
            metrics.append(ROUGEMetric(variant))
        elif metric_name == "meteor":
            metrics.append(METEORMetric())
    return metrics

def load_list_data(gt_path, pred_path=None):
    """Load list data from CSV files, handling different column structures."""
    # Load ground truth data
    gt_data = pd.read_csv(gt_path)
    
    # Process ground truth data - split items into lists and group by panel
    def process_gt_items(items):
        if pd.isna(items) or not isinstance(items, str):
            return []
        return [item.strip() for item in items.split(',') if item.strip()]

    # Group GT data by panel and create lists of objects
    gt_processed = (gt_data.groupby(['subdb', 'comic_no', 'page_no', 'panel_no'])
                   .agg({'items': lambda x: process_gt_items(x.iloc[0])})
                   .reset_index())
    
    if pred_path is None or pred_path == gt_path:
        return gt_processed, gt_processed
    
    # Load and process prediction data
    pred_data = pd.read_csv(pred_path)
    
    # For predictions, take first object from each row and group by panel
    def process_pred_objects(x):
        if pd.isna(x) or not isinstance(x, str):
            return ''
        return x.split(',')[0].strip()
    
    # Process predictions - get first object from each row and group into lists by panel
    pred_processed = (pred_data.assign(main_object=pred_data['objects'].apply(process_pred_objects))
                     .groupby(['subdb', 'comic_no', 'page_no', 'panel_no'])
                     .agg({'main_object': lambda x: list(x)})
                     .reset_index())

    return gt_processed, pred_processed


def compute_bert_similarity(pred_objects: List[str], gt_objects: List[str], tokenizer, model) -> torch.Tensor:
    """Compute BERT-based similarity between predicted and ground truth objects."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        # Batch process the embeddings
        pred_tokens = tokenizer(pred_objects, return_tensors='pt', padding=True, truncation=True)
        gt_tokens = tokenizer(gt_objects, return_tensors='pt', padding=True, truncation=True)
        
        # Move to GPU
        pred_tokens = {k: v.to(device) for k, v in pred_tokens.items()}
        gt_tokens = {k: v.to(device) for k, v in gt_tokens.items()}
        
        pred_embeddings = model(**pred_tokens)['last_hidden_state'][:,0]
        gt_embeddings = model(**gt_tokens)['last_hidden_state'][:,0]
    
    # Normalize embeddings
    pred_norm = pred_embeddings / pred_embeddings.norm(dim=1, keepdim=True)
    gt_norm = gt_embeddings / gt_embeddings.norm(dim=1, keepdim=True)
    
    # Compute similarity matrix
    similarity = torch.mm(pred_norm, gt_norm.t())
    return similarity.cpu()  # Return to CPU for further processing

def match_objects(pred_objects: List[str], gt_objects: List[str], similarity_matrix: torch.Tensor, threshold: float = 0.8) -> List[str]:
    """Match predicted objects to ground truth objects using Hungarian matching."""
    from scipy.optimize import linear_sum_assignment

    # Convert similarity matrix to cost matrix for Hungarian algorithm
    cost_matrix = 1 - similarity_matrix.numpy()
    
    # Find optimal matching
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Create updated predictions list
    updated_preds = pred_objects.copy()
    
    # Replace matched predictions with ground truth if similarity exceeds threshold
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if similarity_matrix[pred_idx, gt_idx] > threshold:
            updated_preds[pred_idx] = gt_objects[gt_idx]
    
    return updated_preds

def evaluate_object_lists(gt_data: pd.DataFrame, pred_data: pd.DataFrame) -> None:
    """Evaluate object lists using BERT similarity and set-based metrics."""
    from transformers import AutoTokenizer, AutoModel
    
    print("Initializing BERT model...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Process in batches
    BATCH_SIZE = 32  # Adjust based on your GPU memory
    
    # Merge panels to evaluate
    panel_keys = ['subdb', 'comic_no', 'page_no', 'panel_no']
    merged_panels = pd.merge(
        gt_data,
        pred_data,
        on=panel_keys,
        how='inner'
    )
    
    # Initialize metrics storage
    thresholds = [0.5, 0.8, 0.9, 0.95, 0.999]
    metrics = {t: {'precision': [], 'recall': [], 'f1': []} for t in thresholds}
    
    # Process in batches
    for i in tqdm(range(0, len(merged_panels), BATCH_SIZE), desc="Processing batches"):
        batch = merged_panels.iloc[i:i+BATCH_SIZE]
        
        for _, row in batch.iterrows():
            gt_objects = row['items']
            pred_objects = row['main_object']
            
            if not gt_objects or not pred_objects:
                continue
                
            similarity_matrix = compute_bert_similarity(pred_objects, gt_objects, tokenizer, model)
            
            for threshold in thresholds:
                updated_preds = match_objects(pred_objects, gt_objects, similarity_matrix, threshold)
                
                # Calculate metrics
                gt_set = set(gt_objects)
                pred_set = set(updated_preds)
                
                intersection = len(gt_set & pred_set)
                precision = intersection / len(pred_set) if pred_set else 0
                recall = intersection / len(gt_set) if gt_set else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics[threshold]['precision'].append(precision)
                metrics[threshold]['recall'].append(recall)
                metrics[threshold]['f1'].append(f1)
    
    # Calculate average scores, per threshold
    avg_precision = { threshold: sum(metrics[threshold]['precision']) / len(metrics[threshold]['precision']) if metrics[threshold]['precision'] else 0 for threshold in thresholds }
    avg_recall = { threshold: sum(metrics[threshold]['recall']) / len(metrics[threshold]['recall']) if metrics[threshold]['recall'] else 0 for threshold in thresholds }
    avg_f1 = { threshold: sum(metrics[threshold]['f1']) / len(metrics[threshold]['f1']) if metrics[threshold]['f1'] else 0 for threshold in thresholds }
    
    print("\n=== Object List Evaluation Scores (BERT-enhanced) ===\n")
    print("Metric\t\t", end="")
    for t in thresholds:
        print(f"{t:5.3f}\t", end="")
    print()
    
    # Print each metric row
    print(f"Precision\t" + "\t".join(f"{avg_precision[t]:5.3f}" for t in thresholds))
    print(f"Recall\t\t" + "\t".join(f"{avg_recall[t]:5.3f}" for t in thresholds)) 
    print(f"F1-Score\t" + "\t".join(f"{avg_f1[t]:5.3f}" for t in thresholds))
    print()


    # Print panel statistics
    print("\nPanel Statistics:")
    print(f"Total panels evaluated: {len(metrics[0.5]['precision'])}")
    print(f"Average objects per panel in GT: {merged_panels['items'].apply(len).mean():.2f}")
    print(f"Average objects per panel in Pred: {merged_panels['main_object'].apply(len).mean():.2f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate captions against ground truth.')
    parser.add_argument('-p', '--pred_file', help='Path to prediction file (optional)', default=None)
    # calculate only NLP metrics
    parser.add_argument('-nlp', '--nlp_only', help='Calculate only NLP metrics', action='store_true')
    # calculate only object list metrics
    parser.add_argument('-arm', '--arm_only', help='Calculate only Attribute Retention Metric', action='store_true')
    args = parser.parse_args()
    
    HOME_DIR = "/home/evivoli/projects/CoMix"
    FILE_DIR = f"{HOME_DIR}/benchmarks/cap_val/eval"

    gt_path = f"{FILE_DIR}/gt_captions.csv"
    gt_list_path = f"{FILE_DIR}/gt_lists.csv"

    if not args.pred_file:
        print("No prediction file provided. Using ground truth for both captions and lists.")
        pred_path = gt_path
        pred_list_path = gt_list_path
    else:
        pred_path = f"{FILE_DIR}/{args.pred_file}_captions.csv"
        pred_list_path = f"{FILE_DIR}/{args.pred_file}_lists.csv" 

    gt_data, pred_data = load_data(gt_path, pred_path)
    missing_gt_df, missing_pred_df = identify_missing_entries(gt_data, pred_data)
    save_missing_instances(missing_gt_df, missing_pred_df, "captions")

    list_gt_data, list_pred_data = load_list_data(gt_list_path, pred_list_path)
    list_missing_gt_df, list_missing_pred_df = identify_missing_entries(list_gt_data, list_pred_data)
    save_missing_instances(list_missing_gt_df, list_missing_pred_df, "lists")

    if missing_gt_df.empty and missing_pred_df.empty:
        print("No missing entries found in either ground truth or predictions.")
    else:
        print("Missing entries found. Details saved to missing_instances.txt.")

    if list_missing_gt_df.empty and list_missing_pred_df.empty:
        print("No missing entries found in either ground truth or predictions.")
    else:
        print("Missing entries found. Details saved to missing_instances.txt.")

    # Evaluate object lists instead of captions
    metrics_names = ["bleu", "bleu1", "bleu2", "bleu3", "bleu4", "rougel", "rouge1", "rouge2", "meteor"]
    if args.nlp_only:
        evaluate_captions(gt_data, pred_data, metrics_names)
    elif args.arm_only:
        evaluate_object_lists(list_gt_data, list_pred_data)
    else:
        evaluate_captions(gt_data, pred_data, metrics_names)
        evaluate_object_lists(list_gt_data, list_pred_data)


if __name__ == "__main__":
    main()