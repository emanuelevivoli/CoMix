import os
from pycocotools.coco import COCO
import torchvision
from tqdm import tqdm
import json
from PIL import Image
import torch

from torchvision.transforms import v2 as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

def collate_fn(batch):
    return tuple(zip(*batch))

from comix.utils.coco import add_image
from comix.utils.constants import COCO_OUTPUT
from comix.utils.data import adapt_faster_bbox
from comix.utils.dataset import FasterRCNNInferenceDataset
from comix.utils import get_image_id, add_annotation

import argparse

# get time for logging
from datetime import datetime

# give me the time ddmmyyyy-hhmmss
dt_string = datetime.now().strftime("%d%m%Y-%H%M%S")

# 4 classes (panel, character, text, face) + background
NUM_CLASSES = 4 + 1

def get_transform(train):
    transforms = []
    transforms.append(T.Resize((1024, 1024)))
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def arg_parser():
    parser = argparse.ArgumentParser(description='Evaluate COCO predictions.')
    # Add all necessary argument definitions here
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-n', '--dataset_name', type=str, default='popmanga', help='Name of the dataset', choices=['comics', 'DCM', 'eBDtheque','popmanga'])
    parser.add_argument('-s', '--split', type=str, default='val', help='Dataset split (e.g., validation, test)', choices=['val', 'validation', 'test', 'all'])
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--input_dir', type=str, default='data/datasets.unify', help='Path to the dataset')
    # parser.add_argument('--input_yolo', type=str, default='data/comix.yolo', help='Path to the YOLO dataset')
    parser.add_argument('--weights_path', type=str, default='benchmarks/weights', help='Path to save the FasterRCNN model')
    parser.add_argument('-wn', '--weights_name', type=str, default='faster_rcnn-c100-best-10052024_092536', help='Name of the YOLOv8 model')
    parser.add_argument('-nw', '--num_workers', type=int, default=4, help='Number of workers for the data loader')
    parser.add_argument('--output_coco', type=str, default='data/predicts.coco', help='Path to save prediction COCO JSON file')
    parser.add_argument('--old', action='store_true', help='Use the old mapping')
    parser.add_argument('--save', type=int, default=None, help='Number of images to save (showing GT and predictions)')
    args = parser.parse_args()
    return args

args = arg_parser()

def get_model_instance_detection(num_classes=NUM_CLASSES):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT) #, pretrained=True)
    num_classes = 5  # 4 classes (panel, character, text, face) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if args.dataset_name == 'Manga109' and args.mode == 'custom':

    val_dataset = FasterRCNNInferenceDataset(
        f'{args.input_dir}/{args.dataset_name}/original/images',
        f'{args.input_dir}/{args.dataset_name}/splits/{args.mode}',
        transform=get_transform(train=False),
        scale=1024,
        split=args.split,
        mode='train',
        db = 'iyyer' if 'iyyer' in args.dataset_name else None
    )
else:
   
    val_dataset = FasterRCNNInferenceDataset(
        f'{args.input_dir}/{args.dataset_name}/images',
        f'{args.input_dir}/{args.dataset_name}/splits',
        transform=get_transform(train=False),
        scale=1024,
        split=args.split,
        db = 'iyyer' if 'iyyer' in args.dataset_name else None
    )

# define validation data loaders
data_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=collate_fn
)

if args.save:
    # save_path = f'out/output_predictions/{args.dataset_name}/{args.weights_name}-{dt_string}'
    os.makedirs(save_path, exist_ok=True)

output_base_path = f'{args.output_coco}/{args.dataset_name}'
    
if 'mix' in args.weights_name:
    weights_subdir = 'faster_mix'
elif 'm109' in args.weights_name:
    weights_subdir = 'faster_m109'
elif 'c100-last' in args.weights_name:
    weights_subdir = 'faster_c100-last'
else:
    weights_subdir = 'faster_c100-best'
    
save_path = os.path.join(output_base_path, weights_subdir)
os.makedirs(save_path, exist_ok=True)

# get the model using our helper function
model = get_model_instance_detection(NUM_CLASSES)
# use the pretrained model
model.load_state_dict(torch.load(f'{args.weights_path}/faster-rcnn/{args.weights_name}.pth'))
model.eval()
# move model to the right device
model.to(device)

iyyer_mapping = [1, 0, 2, 3]
fasterrcnn_mapping = [None, 2, 1, 3, 0]
CLS_MAPPING = {
    0: 1,  # panel
    1: 2,  # character
    2: 4,  # text
    3: 7   # face
}

CLS2COLOR = {
    1: 'green', # panel
    2: 'red', # character
    4: 'blue', # text
    7: 'magenta' # face
}

coco_output = COCO_OUTPUT
FASTERCNN_RESHAPE = (1024, 1024)

annotation_id = 0
with torch.no_grad():
    cpu_device = torch.device("cpu")
    for (batch_imgs, batch_paths) in tqdm(data_loader):
        batch_imgs = list(img.to(device) for img in batch_imgs)
        batch_results = model(batch_imgs)

        batch_results = [{k: v.to(cpu_device) for k, v in t.items()} for t in batch_results]

        res = {}
        for results, to_unpack in zip(batch_results, batch_paths):
            (image_path, comic_no, page_no) = to_unpack
            image_id = get_image_id(comic_no, page_no)
            coco_output = add_image(coco_output, image_id, image_path)
            img_size = Image.open(image_path).size

            # remove bellow 0.5 confidence
            to_keep = results['scores'] > 0.5
            results['scores'] = results['scores'][to_keep]
            results['boxes'] = results['boxes'][to_keep]
            results['labels'] = results['labels'][to_keep]

            for i, (box, score, label) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):
                xyxy = box.cpu().numpy().tolist()
                cls = label.cpu().numpy().item()

                if 'iyyer' in args.weights_path and 'yolo' in args.weights_path:
                    cls = iyyer_mapping[int(cls)]
                elif args.old:
                    cls = fasterrcnn_mapping[int(cls)]
                else:
                    cls = int(cls)-1
                _xywh = adapt_faster_bbox(xyxy, img_size, FASTERCNN_RESHAPE)
                coco_output = add_annotation(coco_output, annotation_id, image_id, CLS_MAPPING[int(cls)], _xywh, 1.0)
                annotation_id += 1

# Save the COCO JSON
output_path = os.path.join(save_path, f'{args.split}.json')

with open(output_path, 'w') as f:
    json.dump(coco_output, f)

if args.save:

    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    i = 0
    for (batch_imgs, batch_paths) in data_loader:
        for to_unpack in enumerate(batch_paths):
            (img_path, comic_no, page_no) = to_unpack

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

            i += 1
            if i >= args.save:
                exit()