import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from comix.utils import MAGI_TARGET_SIZE, YOLO_TARGET_SIZE, DASS_TARGET_SIZE, DINO_TARGET_SIZE
# from torchvision import tv_tensors

class DinoDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_size=DINO_TARGET_SIZE, color=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the csv file, it has the headers: book, page
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.target_size = target_size
        self.color = color
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        import os
        image_path = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 2],  # book
                                str(self.data_frame.iloc[idx, 3]).zfill(3) + '.jpg')  # page


        image = self.read_image_as_np_array(image_path, self.target_size, self.color)
        return image, (image_path, self.data_frame.iloc[idx, 2], self.data_frame.iloc[idx, 3])

    @staticmethod
    def read_image_as_np_array(image_path, target_size, color):
        from PIL import Image, ImageFile, UnidentifiedImageError
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(image_path) as img:
                if not color:
                    img = img.convert("L")
                img = img.convert("RGB")

                # Resize or transpose image if necessary
                if img.width > img.height:
                    img = img.transpose(Image.ROTATE_90)

                # Use ImageResampling.LANCZOS for high-quality downsampling
                img = img.resize(target_size, Image.Resampling.LANCZOS)

                img = np.array(img)
            return img
        except (Image.DecompressionBombError, UnidentifiedImageError):
            print(f"Error loading image at {image_path}. Skipping.")
            return None

class MagiDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_size=MAGI_TARGET_SIZE, color=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the csv file, it has the headers: book, page
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.target_size = target_size
        self.color = color
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        import os
        image_path = os.path.join(self.root_dir, 
                                self.data_frame.iloc[idx, 2],  # book
                                str(self.data_frame.iloc[idx, 3]).zfill(3) + '.jpg')  # page
        

        image = self.read_image_as_np_array(image_path, self.target_size, self.color)
        return image, (image_path, self.data_frame.iloc[idx, 2], self.data_frame.iloc[idx, 3])

    @staticmethod
    def read_image_as_np_array(image_path, target_size, color):
        from PIL import Image, ImageFile, UnidentifiedImageError
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(image_path) as img:
                if not color:
                    img = img.convert("L")
                img = img.convert("RGB")

                # Resize or transpose image if necessary
                if img.width > img.height:
                    img = img.transpose(Image.ROTATE_90)

                # Use ImageResampling.LANCZOS for high-quality downsampling
                img = img.resize(target_size, Image.Resampling.LANCZOS)

                img = np.array(img)
            return img
        except (Image.DecompressionBombError, UnidentifiedImageError):
            print(f"Error loading image at {image_path}. Skipping.")
            return None

class DASSDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_size=DASS_TARGET_SIZE, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the csv file, it has the headers: book, page
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        import cv2
        import os
        image_path = os.path.join(self.root_dir, 
                                self.data_frame.iloc[idx, 2],  # book
                                str(self.data_frame.iloc[idx, 3]).zfill(3) + '.jpg')  # page
        

        img = cv2.imread(image_path)
        h, w, c = img.shape
        if self.transform:
            img, _ = self.transform(img, None, self.target_size)
        scale = min(self.target_size[0] / h, self.target_size[1] / w)
        return img, scale, h, w, (image_path, self.data_frame.iloc[idx, 2], self.data_frame.iloc[idx, 3])

class YoloInferenceDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_size=YOLO_TARGET_SIZE, color=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the csv file, it has the headers: book, page
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.target_size = target_size
        self.color = color
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        import os
        image_path = os.path.join(self.root_dir, 
                                self.data_frame.iloc[idx, 2],  # book
                                str(self.data_frame.iloc[idx, 3]).zfill(3) + '.jpg')  # page
        

        image = self.read_image_as_np_array(image_path, self.target_size, self.color)
        return image, (image_path, self.data_frame.iloc[idx, 2], self.data_frame.iloc[idx, 3])

    @staticmethod
    def read_image_as_np_array(image_path, target_size, color, transpose=True):
        from PIL import Image, ImageFile, UnidentifiedImageError
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(image_path) as img:

                img = img.convert("L" if not color else "RGB")

                # Resize or transpose image if necessary
                if transpose and img.width > img.height:
                    img = img.transpose(Image.ROTATE_90)

                if target_size is not None:
                    # Use ImageResampling.LANCZOS for high-quality downsampling
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # if yolo
                img = np.array(img).transpose(2, 0, 1)
                img = img / 255.0
                
            return img
        except (Image.DecompressionBombError, UnidentifiedImageError):
            print(f"Error loading image at {image_path}. Skipping.")
            return None

class FasterRCNNInferenceDataset(Dataset):
    def __init__(self, images_dir, csv_file, mode=None, split='train', transform=None, target_transform=None, scale=1024, db=None):
        """
        Args:
            images_dir (string): Path to the images folder.
            target_dir (string): Path to the target folder.
            target_size (tuple): Target size for the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the target.
        """
        self.images_dir = images_dir
        self.scale = scale
        self.data_frame = pd.read_csv(f'{csv_file}/{split}.csv')
        self.mode = mode if mode is not None else split
        self.db = db

        self.transform = transform
        # self.target_transform = target_transform

        self.remove_broken_images()

    def __len__(self):
        return len(self.data_frame)

    def remove_broken_images(self):
        import os
        from PIL import Image, UnidentifiedImageError
        df_copy = self.data_frame.copy()
        # check every image and target file if they are broken
        for idx in range(len(df_copy)):
            book_id = df_copy.iloc[idx, 2]
            # check for the images
            if self.db and self.db == 'iyyer':
                page_id = str(df_copy.iloc[idx, 3])
                image_path = os.path.join(self.images_dir, f'{book_id}_{page_id}' + '.jpg')
            else:
                page_id = str(df_copy.iloc[idx, 3]).zfill(3)
                image_path = os.path.join(self.images_dir, book_id, page_id + '.jpg')
            image_broken = True if not os.path.exists(image_path) else False
            if not image_broken:
                try:
                    img = Image.open(image_path)
                    img.verify()
                    if img.width == 0 or img.height == 0:
                        raise ValueError(f"Image at {image_path} has invalid size.")
                except (Image.DecompressionBombError, UnidentifiedImageError):
                    print(f"Error loading image at {image_path}. Removing.")
                    image_broken = True

            if image_broken:
                print(f"Removing broken image at {image_path}")
                self.data_frame.drop(idx, inplace=True)

    def __getitem__(self, idx):
        import os
        book_id = str(self.data_frame.iloc[idx, 2])

        if self.db and self.db == 'iyyer':
            page_id = str(self.data_frame.iloc[idx, 3])
            image_path = os.path.join(self.images_dir, self.mode, f'{book_id}_{page_id}' + '.jpg')
        else:
            page_id = str(self.data_frame.iloc[idx, 3]).zfill(3)
            image_path = os.path.join(self.images_dir, book_id, page_id + '.jpg')

        image = self.read_image(image_path)

        if image is None:
            return None

        if self.transform:
            image = self.transform(image)

        return image, (image_path, book_id, page_id)
    
    def read_image(self, image_path):
        from PIL import Image
        with Image.open(image_path) as image:
            return image.copy()  # This ensures the image data is retained after the file is closed

class SSDInferenceDataset(Dataset):
    def __init__(self, images_dir, csv_file, mode=None, split='train', transforms=None, db=None):
        """
        Args:
            images_dir (string): Path to the images folder.
            target_dir (string): Path to the target folder.
            target_size (tuple): Target size for the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the target.
        """
        self.images_dir = images_dir
        self.data_frame = pd.read_csv(f'{csv_file}/{split}.csv')
        self.mode = mode if mode is not None else split
        self.transforms = transforms
        self.db = db

        self.remove_broken_images()

    def __len__(self):
        return len(self.data_frame)

    def remove_broken_images(self):
        import os
        from PIL import Image, UnidentifiedImageError
        df_copy = self.data_frame.copy()
        # check every image and target file if they are broken
        for idx in range(len(df_copy)):
            book_id = df_copy.iloc[idx, 2]
            # check for the images
            if self.db and self.db == 'iyyer':
                page_id = str(df_copy.iloc[idx, 3])
                image_path = os.path.join(self.images_dir, self.mode, f'{book_id}_{page_id}' + '.jpg')
            else:
                page_id = str(df_copy.iloc[idx, 3]).zfill(3)
                image_path = os.path.join(self.images_dir, book_id, page_id + '.jpg')
            image_broken = True if not os.path.exists(image_path) else False
            if not image_broken:
                try:
                    img = Image.open(image_path)
                    img.verify()
                    if img.width == 0 or img.height == 0:
                        raise ValueError(f"Image at {image_path} has invalid size.")
                except (Image.DecompressionBombError, UnidentifiedImageError):
                    print(f"Error loading image at {image_path}. Removing.")
                    image_broken = True

            if image_broken:
                print(f"Removing broken image at {image_path}")
                self.data_frame.drop(idx, inplace=True)

    def __getitem__(self, idx):
        import os
        book_id = str(self.data_frame.iloc[idx, 2])

        if self.db and self.db == 'iyyer':
            page_id = str(self.data_frame.iloc[idx, 3])
            image_path = os.path.join(self.images_dir, self.mode, f'{book_id}_{page_id}' + '.jpg')
        else:
            page_id = str(self.data_frame.iloc[idx, 3]).zfill(3)
            image_path = os.path.join(self.images_dir, book_id, page_id + '.jpg')

        image = self.read_image(image_path)

        if image is None:
            return None
        
        image = torch.from_numpy(np.array(image))

        return image, (image_path, book_id, page_id)
    
    def read_image(self, image_path):
        from PIL import Image
        with Image.open(image_path) as image:
            return image.copy()  # This ensures the image data is retained after the file is closed

