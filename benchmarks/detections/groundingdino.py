import os
import json
import argparse
from tqdm import tqdm
import torch
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
from torch.utils.data import DataLoader
from PIL import Image

torch.set_grad_enabled(False)  # Disable gradient computation globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from comix.utils import (
    get_image_id, add_image, add_annotation, 
    COCO_OUTPUT, DINO_TARGET_SIZE, 
    adapt_dino_bbox, DinoDataset
)

parser = argparse.ArgumentParser(description='Evaluate COCO predictions.')
parser.add_argument('-n', '--dataset_name', type=str, default='popmanga', help='Name of the dataset')
parser.add_argument('-s', '--split', type=str, default='val', help='Dataset split (e.g., val, test)', choices=['all','validation', 'test', 'train', 'val'])
parser.add_argument('--save', type=int, default=None, help='Number of images to save (showing GT and predictions)')
parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--input_unify', type=str, default='data/datasets.unify', help='Path to unified XML folders')
parser.add_argument('--output_coco', type=str, default='data/predicts.coco', help='Path to save prediction COCO JSON file')
args = parser.parse_args()

if args.save:
    save_path = f'out/output_predictions/{args.dataset_name}/GD'
    os.makedirs(save_path, exist_ok=True)


dataset = DinoDataset(csv_file=f'{args.input_unify}/{args.dataset_name}/splits/{args.split}.csv',
                        root_dir=f'{args.input_unify}/{args.dataset_name}/images',
                        target_size=DINO_TARGET_SIZE,
                        color=True)

data_loader = DataLoader(
  dataset, 
  batch_size=args.batch_size,
  num_workers=4,
  shuffle=False,
  pin_memory=True  # Enable pinning memory
)

#MODEL
processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device).eval()

coco_output = COCO_OUTPUT

def process(batch, text_type):
  postprocessed_outputs = []
  with torch.no_grad():
    inputs = processor(images=batch, text=text_type, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([img.shape[0:2] for img in batch])
    postprocessed_outputs = processor.post_process_grounded_object_detection(outputs,
                                                                input_ids=inputs.input_ids,
                                                                target_sizes=target_sizes,
                                                                box_threshold=0.3,
                                                                text_threshold=0.1)

  return postprocessed_outputs


def unify_results(batch_results, new_name):
    new_batch = []
    for res in batch_results:
        scores = [score.item() for score in res["scores"]]
        labels = [new_name] * len(scores)
        boxes = [[b.item() for b in box] for box in res["boxes"]]
        
        results = {
            'scores': scores,
            'labels': labels,
            'boxes': boxes
        }
        new_batch.append(results)
    return new_batch



texts={
  'panel': "panel . comic panel . frame .",
  'character': "character . person . boy . girl . student . woman . man . animal . human . individual.",
  'text': "text . script . writing . printed text . handwritten text .",
  'face': "face . facial expression . "
}


annotation_id = 1
with torch.no_grad():
    for (batch_imgs, batch_paths) in tqdm(data_loader):
      batch_imgs = [batch_imgs[i].cpu().numpy() for i in range(len(batch_imgs))]
      
      cls2results = {}
      for cls_str, text in texts.items():
        res = process(batch_imgs, [text]*len(batch_imgs))
        new_res = unify_results(res, cls_str)
        cls2results[cls_str] = new_res

      for i , (img_path, comic_no, page_no) in enumerate(zip(*batch_paths)):
        image_id = get_image_id(comic_no, page_no)
        coco_output = add_image(coco_output, image_id, img_path)
        image_size = Image.open(img_path).size

        #Panel
        res_panel_imm = cls2results['panel'][i]
        for bbox, score in zip(res_panel_imm['boxes'], res_panel_imm['scores']):
          bbox = adapt_dino_bbox(bbox, image_size)
          coco_output = add_annotation(coco_output, annotation_id, image_id, 1, bbox, score)
          annotation_id += 1

        #Character
        res_chara_imm = cls2results['character'][i]
        for bbox, score in zip(res_chara_imm['boxes'], res_chara_imm['scores']):
          bbox = adapt_dino_bbox(bbox, image_size)
          coco_output = add_annotation(coco_output, annotation_id, image_id, 2, bbox, score)
          annotation_id += 1

        #Text
        res_text_imm = cls2results['text'][i]
        for bbox, score in zip(res_text_imm['boxes'], res_panel_imm['scores']):
          bbox = adapt_dino_bbox(bbox, image_size)
          coco_output = add_annotation(coco_output, annotation_id, image_id, 4, bbox, score)
          annotation_id += 1

        #Face
        res_face_imm = cls2results['face'][i]
        for bbox, score in zip(res_face_imm['boxes'], res_face_imm['scores']):
          bbox = adapt_dino_bbox(bbox, image_size)
          coco_output = add_annotation(coco_output, annotation_id, image_id, 7, bbox, score)
          annotation_id += 1

        #print(coco_output)


        #break
# Create the output directory if it doesn't exist
output_path = f'{args.output_coco}/{args.dataset_name}/grounding'
os.makedirs(output_path, exist_ok=True)

# Save the COCO JSON
output_path = f'{output_path}/{args.split}.json'
with open(output_path, 'w') as f:
    json.dump(coco_output, f)
