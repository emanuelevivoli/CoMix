MAGI_TARGET_SIZE = (3627, 5311)
YOLO_TARGET_SIZE = (1024, 1024)
DASS_TARGET_SIZE = (2048, 2048)
DINO_TARGET_SIZE = (800, 1200)

CLS2COLOR = {
    1: 'green', # panel
    2: 'red', # character
    4: 'blue', # text
    7: 'magenta' # face
}

COCO_OUTPUT = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "panel"},
        {"id": 2, "name": "character"},
        {"id": 3, "name": "balloon"},
        {"id": 4, "name": "text"},
        {"id": 5, "name": "onomatopoeia"},
        {"id": 7, "name": "face"},
        {"id": 6, "name": "link_sbsc"}
    ]
}

LABEL2ID = {
    "panel": 1,
    "character": 2,
    "balloon": 3,
    "text": 4,
    "onomatopoeia": 5,
    "face": 7,
    "link_sbsc": 6
}

DETECTION_CATEGORIES = {
    "panel": 1,
    "character": 2,
    # "balloon": 3,
    "text": 3,
    # "onomatopoeia": 5,
    "face": 4,
}

KEYPOINT_CATEGORIES = {
    "link_sbsc": {
        "id": 6,
        "keypoints": ["balloon", "character"],
        "skeleton": [[0, 1]]
    }
} 

DB_MAPPING = {
    'ciyyer': 'ciyyer',
    'comics': 'c100',
    # 'dcm': 'dcm', #? reserved for val and test
    # 'eBDtheque': 'ebd', #? reserved for val and test
    'Manga109': 'm109',
    # 'popmanga': 'pmg', #? dev set not available
    'mix': 'mix'
}