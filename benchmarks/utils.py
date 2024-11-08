import torch
import numpy as np

def batched_xywh2xyxy(batch):
    # batch: (N, 4) tensor
    xy = batch[:, :2]
    wh = batch[:, 2:]
    return torch.cat([xy - wh / 2, xy + wh / 2], dim=1)

def batched_xyxy2ccwh(batch):
    # batch: (N, 4) tensor
    xy = (batch[:, :2] + batch[:, 2:]) / 2
    wh = batch[:, 2:] - batch[:, :2]
    return torch.cat([xy, wh], dim=1)

def xyxy2ccwh(bbox):
    # bbox: (4,) tensor
    xy = (bbox[:2] + bbox[2:]) / 2
    wh = bbox[2:] - bbox[:2]
    # use ccwh format in numpy
    return np.array([xy[0].item(), xy[1].item(), wh[0].item(), wh[1].item()])

def xywh2ccwh(bbox, img_w, img_h):
    # The COCO box format is [top left x, top left y, width, height]
    bbox[:2] += bbox[2:] / 2  # xy top-left corner to center
    bbox[[0, 2]] /= img_w  # normalize x
    bbox[[1, 3]] /= img_h  # normalize y
    return bbox
                    
def scale_boxes(boxes, scale=1024):
    return boxes * scale

def scale_boxes_custom(boxes, scale=(1024, 1024)):
    # scale to H, W
    boxes[:, [0,2]] *= scale[1]
    boxes[:, [1,3]] *= scale[0]
    return boxes

def ssd_custom_collate(batch, scale=1024):
    # discard None images
    batch = [x for x in batch if None not in x]
    batch_images, batch_targets = list(zip(*batch))
    batch_images = torch.stack(batch_images)

    batch_targets = [{
        'boxes': scale_boxes(batched_xywh2xyxy(torch.from_numpy(np.array(targets)[:,1:].astype(float))), scale=scale),
        'labels': torch.from_numpy(np.array(targets)[:,0].astype(int)),
    } for targets in batch_targets]

    return batch_images, batch_targets

def another_custom_collate(batch):
    # discard None images
    batch = [x for x in batch if None not in x]
    batch_images, batch_targets = list(zip(*batch))
    images = torch.stack(batch_images)
    # pad all targets['boxes'] to the same length
    max_boxes = max([len(targets['boxes']) for targets in batch_targets])
    for i, targets in enumerate(batch_targets):
        targets['boxes'] = torch.cat([targets['boxes'], torch.zeros(max_boxes - len(targets['boxes']), 4)])
        targets['labels'] = torch.cat([targets['labels'], -1*torch.ones(max_boxes - len(targets['labels']))])
        batch_targets[i] = targets

    return images, batch_targets