import datetime
import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Evaluate COCO predictions.')
parser.add_argument('-n', '--dataset_name', type=str, default='popmanga', help='Name of the dataset')
parser.add_argument('-s', '--split', type=str, default='val', help='Dataset split (e.g., validation, test)', choices=['validation', 'test', 'all', 'val'])
parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--save', type=int, default=None, help='Number of images to save (showing GT and predictions)')
parser.add_argument('--input_dir', type=str, default='data/datasets.unify', help='Path to splits and images')
parser.add_argument('--output_coco', type=str, default='data/predicts.coco', help='Path to save prediction COCO JSON file')
parser.add_argument('--weights_path', type=str, default='benchmarks/weights', help='Path to the YOLOv8 model')
parser.add_argument('-wn', '--weights_name', type=str, default='yolov8x-best', help='Name of the YOLOv8 model')
parser.add_argument('--new', action='store_true', help='Use the new adapt_yolo_bbox function')
args = parser.parse_args()


from comix.utils import COCO_OUTPUT
coco_output = COCO_OUTPUT
# give me the time ddmmyyyy-hhmmss
dt_string = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")

from comix.utils import YoloInferenceDataset

# Dataset
dataset = YoloInferenceDataset(csv_file=f'{args.input_dir}/{args.dataset_name}/splits/{args.split}.csv',
                               root_dir=f'{args.input_dir}/{args.dataset_name}/images',
                        # label_dir=f'{args.input_dir
                        # }/{args.dataset_name}/labels',
                        target_size=(1024, 1024), color=True)

# DataLoader
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


if args.save:
    save_path = f'out/output_predictions/{args.dataset_name}/{args.weights_name}-{dt_string}'
    os.makedirs(save_path, exist_ok=True)

# Load the YOLOv8 model
model = YOLO(task='detect', model=f'{args.weights_path}/yolov8/{args.weights_name}.pt', verbose=False)  # pretrained YOLOv8n model
model.model.eval()
# Assuming model, test_dataset, and data_loader are already defined

iyyer_mapping = [1, 0, 2, 3]

CLS_MAPPING = {
    0: 1,  # panel
    1: 2,  # character
    2: 4,  # text
    3: 7   # face
}

annotation_id = 1
with torch.no_grad():

    from comix.utils import add_image, add_annotation, get_image_id
    from comix.utils import adapt_yolo_bbox

    for (batch_imgs, batch_paths) in tqdm(data_loader):
        results = model(batch_imgs, verbose=False)

        for i, (result, (image_path, comic_no, page_no)) in enumerate(zip(results, zip(*batch_paths))):
            image_id = get_image_id(comic_no, page_no)
            coco_output = add_image(coco_output, image_id, image_path)

            img_size = Image.open(image_path).size

            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()

            for _xyxy, _cls in zip(xyxy, cls):
                _xywh = adapt_yolo_bbox(_xyxy, img_size)
                if 'iyyer' in args.weights_name and 'yolo' in args.weights_name:
                    _cls = iyyer_mapping[int(_cls)]
                coco_output = add_annotation(coco_output, annotation_id, image_id, CLS_MAPPING[int(_cls)], _xywh, 1.0)
                annotation_id += 1


output_base_path = f'{args.output_coco}/{args.dataset_name}'
    
if 'mix' in args.weights_name:
    weights_subdir = 'yolo_mix'
elif 'm109' in args.weights_name:
    weights_subdir = 'yolo_m109'
elif 'c100' in args.weights_name:
    weights_subdir = 'yolo_c100'
else:
    weights_subdir = 'yolo-best'
    
save_path = os.path.join(output_base_path, weights_subdir)
os.makedirs(save_path, exist_ok=True)


output_path = os.path.join(save_path, f'{args.split}.json')


with open(output_path, 'w') as f:
    json.dump(coco_output, f)

count = 0

if args.save:

    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for (batch_imgs, batch_paths) in data_loader:
        for img_path, comic_no, page_no in zip(*batch_paths):
            image_id = get_image_id(comic_no, page_no)
            img = Image.open(img_path)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(img)

            for ann in coco_output['annotations']:
                if ann['image_id'] == image_id:
                    bbox = ann['bbox']
                    cat_id = ann['category_id']
                    rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=CLS2COLOR[cat_id], facecolor='none')
                    ax.add_patch(rect)

            plt.axis('off')
            plt.savefig(f'{save_path}/{image_id}.png')
            plt.close('all')
            
            count += 1
            if count >= args.save:
                exit()
