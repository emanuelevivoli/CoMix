import os
import json
import time

from pycocotools.coco import COCO
from collections import defaultdict
from PIL import Image
from pathlib import Path

class CustomCOCO(COCO):
    def __init__(self, annotation_file=None, remove_cats=[]):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            for remove_cat in remove_cats:
                dataset['categories'] = [cat for cat in dataset['categories'] if cat['id'] != remove_cat]
                dataset['annotations'] = [ann for ann in dataset['annotations'] if ann['category_id'] != remove_cat]
            self.dataset = dataset
            self.createIndex()

    def removeCats(self, remove_cats):
        for remove_cat in remove_cats:
            self.dataset['categories'] = [cat for cat in self.dataset['categories'] if cat['id'] != remove_cat]
            self.dataset['annotations'] = [ann for ann in self.dataset['annotations'] if ann['category_id'] != remove_cat]
        self.createIndex()

# Helper function to add an image to the COCO dataset
def add_image(coco_output, image_id, file_path):
    image = Image.open(file_path)
    # get the last two part of the path
    file_name = os.path.join(*Path(file_path).parts[-2:])
    width, height = image.size
    coco_output["images"].append({
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name
    })
    return coco_output

# Helper function to add an annotation to the COCO dataset
def add_annotation(coco_output, annotation_id, image_id, category_id, bbox, score):
    x_min, y_min, width, height = bbox
    coco_output["annotations"].append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x_min, y_min, width, height],
        "area": width * height,
        "iscrowd": 0,
        "score": score
    })
    return coco_output