from comix.utils import MAGI_TARGET_SIZE, YOLO_TARGET_SIZE, DINO_TARGET_SIZE

def adapt_dino_bbox(bbox, original_size, reshaped_size=DINO_TARGET_SIZE):
    """
    Adjust a single bounding box annotation to the original image size.

    :param bbox: Bounding box annotation in the format (x1, y1, x2, y2).
    :param original_size: Tuple of the original image size (width, height).
    :param reshaped_size: Tuple of the reshaped image size (width, height).
    :return: Adjusted bounding box.
    """
    x1, y1, x2, y2 = bbox

    o_width, o_height = original_size
    r_width, r_height = reshaped_size

    # Determine if the image was rotated based on original dimensions
    rotated = o_width > o_height

    # Scale factors - note the inversion in scale factors if rotated
    scale_x = o_width / r_height if rotated else o_width / r_width
    scale_y = o_height / r_width if rotated else o_height / r_height

    # Adjust for rotation
    if rotated:
        # Rotate 90 degrees clockwise: swap and invert y coordinates
        x1_new, y1_new = r_height - y2, x1
        x2_new, y2_new = r_height - y1, x2

        # Scale bbox back to original size
        x1, y1 = x1_new * scale_x, y1_new * scale_y
        x2, y2 = x2_new * scale_x, y2_new * scale_y
    else:
        # Scale bbox back to original size without rotation
        x1, y1 = x1 * scale_x, y1 * scale_y
        x2, y2 = x2 * scale_x, y2 * scale_y

    return [x1, y1, x2 - x1, y2 - y1]

def adapt_magi_bbox(bbox, original_size, reshaped_size=MAGI_TARGET_SIZE):
    """
    Adjust a single bounding box annotation to the original image size.

    :param bbox: Bounding box annotation in the format (x1, y1, x2, y2).
    :param original_size: Tuple of the original image size (width, height).
    :param reshaped_size: Tuple of the reshaped image size (width, height).
    :return: Adjusted bounding box.
    """
    x1, y1, x2, y2 = bbox

    o_width, o_height = original_size
    r_width, r_height = reshaped_size

    # Determine if the image was rotated based on original dimensions
    rotated = o_width > o_height

    # Scale factors - note the inversion in scale factors if rotated
    scale_x = o_width / r_height if rotated else o_width / r_width
    scale_y = o_height / r_width if rotated else o_height / r_height

    # Adjust for rotation
    if rotated:
        # Rotate 90 degrees clockwise: swap and invert y coordinates
        x1_new, y1_new = r_height - y2, x1
        x2_new, y2_new = r_height - y1, x2

        # Scale bbox back to original size
        x1, y1 = x1_new * scale_x, y1_new * scale_y
        x2, y2 = x2_new * scale_x, y2_new * scale_y
    else:
        # Scale bbox back to original size without rotation
        x1, y1 = x1 * scale_x, y1 * scale_y
        x2, y2 = x2 * scale_x, y2 * scale_y

    return [x1, y1, x2 - x1, y2 - y1]

def adapt_yolo_bbox(bbox, original_size, reshaped_size=YOLO_TARGET_SIZE):
    """
    Adjust a single bounding box annotation to the original image size.

    :param bbox: Bounding box annotation in the format (x1, y1, x2, y2).
    :param original_size: Tuple of the original image size (width, height).
    :param reshaped_size: Tuple of the reshaped image size (width, height).
    :return: Adjusted bounding box.
    """
    x1, y1, x2, y2 = bbox

    o_width, o_height = original_size
    r_width, r_height = reshaped_size

    # Determine if the image was rotated based on original dimensions
    rotated = o_width > o_height

    # Scale factors - note the inversion in scale factors if rotated
    scale_x = o_width / r_height if rotated else o_width / r_width
    scale_y = o_height / r_width if rotated else o_height / r_height

    # Adjust for rotation
    if rotated:
        # Rotate 90 degrees clockwise: swap and invert y coordinates
        x1_new, y1_new = r_height - y2, x1
        x2_new, y2_new = r_height - y1, x2

        # Scale bbox back to original size
        x1, y1 = x1_new * scale_x, y1_new * scale_y
        x2, y2 = x2_new * scale_x, y2_new * scale_y
    else:
        # Scale bbox back to original size without rotation
        x1, y1 = x1 * scale_x, y1 * scale_y
        x2, y2 = x2 * scale_x, y2 * scale_y

    return [x1, y1, x2 - x1, y2 - y1]

def scale_bbox(preds, scale, sizes):
    import torch

    preds[:,:4] /= scale
    preds[:,0]  = torch.max(preds[:,0], torch.zeros(preds.shape[0]))
    preds[:,1]  = torch.max(preds[:,1], torch.zeros(preds.shape[0]))
    preds[:,2]  = torch.min(preds[:,2], torch.zeros(preds.shape[0]).fill_(sizes[1]))
    preds[:,3]  = torch.min(preds[:,3], torch.zeros(preds.shape[0]).fill_(sizes[0]))
    scores      = preds[:,4]

    return preds, scores

def adapt_yolox_bbox(bbox, original_size=None, reshaped_size=None):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]

def adapt_faster_bbox(bbox, original_size, reshaped_size):
    x1, y1, x2, y2 = bbox

    o_width, o_height = original_size
    r_width, r_height = reshaped_size

    # Determine if the image was rotated based on original dimensions
    rotated = o_width > o_height

    # Scale factors - note the inversion in scale factors if rotated
    scale_x = o_width / r_height if rotated else o_width / r_width
    scale_y = o_height / r_width if rotated else o_height / r_height

    # Adjust for rotation
    if rotated:
        # Rotate 90 degrees clockwise: swap and invert y coordinates
        x1_new, y1_new = r_height - y2, x1
        x2_new, y2_new = r_height - y1, x2

        # Scale bbox back to original size
        x1, y1 = x1_new * scale_x, y1_new * scale_y
        x2, y2 = x2_new * scale_x, y2_new * scale_y
    else:
        # Scale bbox back to original size without rotation
        x1, y1 = x1 * scale_x, y1 * scale_y
        x2, y2 = x2 * scale_x, y2 * scale_y

    return [x1, y1, x2 - x1, y2 - y1]