import os
import json
import argparse
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoModel
from torch.utils.data import DataLoader
from comix.utils import get_image_id, add_image, add_annotation, COCO_OUTPUT, MagiDataset, MAGI_TARGET_SIZE, adapt_magi_bbox

parser = argparse.ArgumentParser(description='Evaluate COCO predictions.')
parser.add_argument('-n', '--dataset_name', default='popmanga', help='Name of the dataset')
parser.add_argument('-s', '--split', type=str, default='val', help='Dataset split (e.g., val, test)', choices=['val', 'test'])
parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--save', type=int, default=None, help='Number of images to save (showing GT and predictions)')
parser.add_argument('--input_unify', type=str, default='data/datasets.unify', help='Path to unified XML folders')
parser.add_argument('--output_coco', type=str, default='data/predicts.coco', help='Path to save prediction COCO JSON file')
args = parser.parse_args()

# Dataset
dataset = MagiDataset(csv_file=f'{args.input_unify}/{args.dataset_name}/splits/{args.split}.csv',
                        root_dir=f'{args.input_unify}/{args.dataset_name}/images',
                        target_size=MAGI_TARGET_SIZE,
                        color=True)

# DataLoader
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).cuda()

# Assuming model, test_dataset, and data_loader are already defined
annotation_id = 1

if args.save:
    save_path = f'out/output_predictions/{args.dataset_name}/magi'
    os.makedirs(save_path, exist_ok=True)

coco_output = COCO_OUTPUT
model = model.eval()
with torch.no_grad():
    for (batch_imgs, batch_paths) in tqdm(data_loader):
        batch_imgs = [batch_imgs[i].cpu().numpy() for i in range(len(batch_imgs))]
        results = model.predict_detections_and_associations(batch_imgs)

        for i, (img_path, comic_no, page_no) in enumerate(zip(*batch_paths)):
            image_id = get_image_id(comic_no, page_no)
            coco_output = add_image(coco_output, image_id, img_path)
            image_size = Image.open(img_path).size
            
            result = results[i]
            
            # Characters
            for bbox, score in zip(result['characters'], result['character_scores']):
                bbox = adapt_magi_bbox(bbox, image_size)
                coco_output = add_annotation(coco_output, annotation_id, image_id, 2, bbox, score)
                annotation_id += 1

            # Panels
            for bbox, score in zip(result['panels'], result['panel_scores']):
                bbox = adapt_magi_bbox(bbox, image_size)
                coco_output = add_annotation(coco_output, annotation_id, image_id, 1, bbox, score)
                annotation_id += 1

            # Texts
            for bbox, score in zip(result['texts'], result['text_scores']):
                bbox = adapt_magi_bbox(bbox, image_size)
                coco_output = add_annotation(coco_output, annotation_id, image_id, 4, bbox, score)
                annotation_id += 1

            if args.save:
                if i >= args.save:
                    continue
                model.visualise_single_image_prediction(batch_imgs[i], results[i], filename=f"out/output_predictions/{args.dataset_name}/magi/image_{i}.png")
    

# Create the output directory if it doesn't exist
output_path = f'{args.output_coco}/{args.dataset_name}/magi'
os.makedirs(output_path, exist_ok=True)

# Save the COCO JSON
output_path = f'{output_path}/{args.split}.json'
with open(output_path, 'w') as f:
    json.dump(coco_output, f)

if args.save:

    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for (batch_imgs, batch_paths) in data_loader:
        for i, (img_path, comic_no, page_no) in enumerate(zip(*batch_paths)):
            if i >= args.save:
                break
            image_id = get_image_id(comic_no, page_no)

            img = Image.open(img_path)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(img)

            for ann in coco_output['annotations']:
                if ann['image_id'] == image_id:
                    bbox = ann['bbox']
                    rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

            plt.axis('off')
            plt.savefig(f'{save_path}/{image_id}.png')
            plt.close('all')