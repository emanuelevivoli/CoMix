import os
import copy
import cv2
import json
import argparse
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from comix.utils import get_image_id, add_image, add_annotation, COCO_OUTPUT, DASSDataset, adapt_yolox_bbox, scale_bbox
from modules.DASS_Det_Inference.dass_det.models.yolox import YOLOX
from modules.DASS_Det_Inference.dass_det.models.yolo_head import YOLOXHead
from modules.DASS_Det_Inference.dass_det.models.yolo_head_stem import YOLOXHeadStem
from modules.DASS_Det_Inference.dass_det.models.yolo_pafpn import YOLOPAFPN
from modules.DASS_Det_Inference.dass_det.data.data_augment import ValTransform
from modules.DASS_Det_Inference.dass_det.utils import postprocess, vis
import torch

parser = argparse.ArgumentParser(description='Evaluate COCO predictions.')
parser.add_argument('-n', '--dataset_name', type=str, default='popmanga', help='Name of the dataset')
parser.add_argument('-s', '--split', type=str, default='val', help='Dataset split (e.g., val, test)', choices=['val', 'test'])
parser.add_argument('--save', type=int, default=None, help='Number of images to save (showing GT and predictions)')
parser.add_argument('--input_unify', type=str, default='data/datasets.unify', help='Path to unified XML folders')
parser.add_argument('--output_coco', type=str, default='data/predicts.coco', help='Path to save prediction COCO JSON file')
parser.add_argument('--model_size', type=str, default='xl', help='Model size', choices=['xs', 'xl'])
parser.add_argument('--weights_path', type=str, default='benchmarks/weights/dass', help='Path to the YOLOv8 model')
parser.add_argument('-pd', '--pretrained_data', type=str, default='mixdata', help='Pretrained data for the model', choices=['m109', 'dcm', 'mixdata'])
args = parser.parse_args()

# Dataset
val_transform = ValTransform()
resize_size = (1024, 1024)
dataset = DASSDataset(csv_file=f'{args.input_unify}/{args.dataset_name}/splits/{args.split}.csv',
                        root_dir=f'{args.input_unify}/{args.dataset_name}/images',
                        target_size=resize_size,
                        transform=val_transform)

# DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

model_path = f"{args.weights_path}/{args.model_size}_{args.pretrained_data}_finetuned_stage3.pth"
nms_thold   = 0.4
conf_thold  = 0.65

if args.model_size == "xs":
    depth, width = 0.33, 0.375
elif args.model_size == "xl":
    depth, width = 1.33, 1.25

# mode=0 for face and body detection
# mode=1 for face
# mode=2 for body
MODE = 0
FACE_INDEX = 7
BODY_INDEX = 2
model = YOLOX(backbone=YOLOPAFPN(depth=depth, width=width),
              head_stem=YOLOXHeadStem(width=width),
              face_head=YOLOXHead(1, width=width),
              body_head=YOLOXHead(1, width=width))

d = torch.load(model_path, map_location=torch.device('cpu'))
if "teacher_model" in d.keys():
    model.load_state_dict(d["teacher_model"])
else:
    model.load_state_dict(d["model"])
model = model.eval().cuda()
del d
#----------------------------------------

# Assuming model, test_dataset, and data_loader are already defined
annotation_id = 1

if args.save:
    save_path = f'out/output_predictions/{args.dataset_name}/dass-{args.pretrained_data}'
    os.makedirs(save_path, exist_ok=True)

coco_output = COCO_OUTPUT
model = model.eval()
with torch.no_grad():
    for b, (batch_imgs, batch_scale, batch_h, batch_w, batch_paths) in enumerate(tqdm(data_loader)):
        # Move the batch to the GPU
        batch_imgs = batch_imgs.cuda().float()
        face_preds, body_preds = model(batch_imgs, mode=MODE)
        face_preds = postprocess(face_preds, 1, conf_thold, nms_thold)
        body_preds = postprocess(body_preds, 1, conf_thold, nms_thold)

        for i, (img_path, comic_no, page_no) in enumerate(zip(*batch_paths)):
            image_id = get_image_id(comic_no, page_no)
            coco_output = add_image(coco_output, image_id, img_path)

            scale = batch_scale[i]
            size = (batch_h[i], batch_w[i])

            if face_preds[i] is not None:
                face_preds_i = face_preds[i].cpu() 
            else:
                print(f'No Face in {comic_no} - {page_no}')
                face_preds_i = torch.empty(0, 5)

            if body_preds[i] is not None:
                body_preds_i = body_preds[i].cpu()
            else:
                print(f'No Char in {comic_no} - {page_no}')
                body_preds_i = torch.empty(0, 5)               

            preds = torch.cat([face_preds_i, body_preds_i], dim=0)
            classes = torch.cat([FACE_INDEX*torch.ones(face_preds_i.shape[0]), BODY_INDEX*torch.ones(body_preds_i.shape[0])])
            preds, scores = scale_bbox(preds, scale, size)
            
            # Face
            for pred, cls in zip(preds, classes):
                (x1, y1, x2, y2, _score) = pred.numpy()
                _xywh = adapt_yolox_bbox([x1, y1, x2, y2])
                coco_output = add_annotation(coco_output, annotation_id, image_id, int(cls.item()), [int(el) for el in _xywh], _score.item())
                annotation_id += 1

            if args.save:
                if b >= args.save:
                    continue
                p_img = cv2.imread(img_path)[:,:,::-1]
                # change 2 to 0 and 7 to 1
                fake_classes = classes.clone()
                fake_classes[classes == BODY_INDEX] = 0
                fake_classes[classes == FACE_INDEX] = 1
                save_img = Image.fromarray(vis(copy.deepcopy(p_img), preds[:,:4], scores, fake_classes, conf=0.0, class_names=["Body", "Face"]))
                filename = f"out/output_predictions/{args.dataset_name}/dass-{args.pretrained_data}/image_{b}.png"
                save_img.save(filename)

# Create the output directory if it doesn't exist
output_path = f'{args.output_coco}/{args.dataset_name}/dass-{args.pretrained_data}/'
os.makedirs(output_path, exist_ok=True)

# Save the COCO JSON
output_path = f'{output_path}/{args.split}.json'
with open(output_path, 'w') as f:
    json.dump(coco_output, f)

print(f'Output saved to {output_path}')

if args.save:

    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for i, (batch_imgs, batch_scale, batch_h, batch_w, batch_paths) in enumerate(data_loader):
        for (img_path, comic_no, page_no) in zip(*batch_paths):
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