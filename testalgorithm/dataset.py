import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class Flickr8kDataset(Dataset):
    """
    Flickr8k dataset class.
    Handles loading images and their associated captions.
    """

    def __init__(
        self,
        data_dir,
        caption_file,
        img_size=(224, 224),
        preproc=None,
    ):
        """
        Flickr8k dataset initialization.
        
        Args:
            data_dir (str): Root directory where images are stored.
            caption_file (str): Path to the captions file (Flickr8k.token.txt).
            img_size (int): Target image size after pre-processing.
            preproc: Optional preprocessing strategy (data augmentation).
        """
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size[:2]
        self.preproc = preproc

        # Load the captions from the caption file
        self.captions = self._load_captions(caption_file)

        # Get the list of image filenames
        self.image_filenames = list(self.captions.keys())

    def _load_captions(self, caption_file):
        """
        Load captions from the Flickr8k.token.txt file.
        
        Args:
            caption_file (str): Path to the caption file.
        
        Returns:
            dict: A dictionary with image filenames as keys and list of captions as values.
        """
        captions = {}
        with open(caption_file, 'r') as f:
            for line in f:
                line = line.strip()
                img_filename, caption = line.split('\t')
                img_filename = img_filename.split('#')[0]
                if img_filename not in captions:
                    captions[img_filename] = []
                captions[img_filename].append(caption)
        return captions

    @property
    def input_dim(self):
        """
        Returns the dimension of the images.
        """
        return self.img_size

    def __len__(self):
        """
        Returns the length of the dataset, i.e., the number of images.
        """
        return len(self.image_filenames)

    def pull_item(self, index):
        """
        Retrieve the image and its corresponding captions from the dataset.
        
        Args:
            index (int): Index of the image.
        
        Returns:
            img (numpy.ndarray): Loaded image.
            captions (list): List of captions for the image.
        """
        img_filename = self.image_filenames[index]
        img_path = os.path.join(self.data_dir, img_filename)

        # Load the image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found at {img_path}")

        # Get the captions for the image
        captions = self.captions[img_filename]

        return img, captions

    def resize_getitem(getitem_fn):
        """
        Decorator method to enable dynamic resizing of input images.
        """
        def wrapper(self, index):
            if not isinstance(index, int):
                has_dim = True
                self.img_size = index[0]
                index = index[1]
            else:
                has_dim = False

            ret_val = getitem_fn(self, index)

            if has_dim:
                del self.img_size

            return ret_val

        return wrapper

    @resize_getitem
    def __getitem__(self, index):
        """
        Get the image and captions for a given index and apply preprocessing if necessary.
        
        Args:
            index (int): Index of the image.
        
        Returns:
            img (numpy.ndarray): Preprocessed image.
            captions (list): Corresponding captions.
        """
        img, captions = self.pull_item(index)

        # Resize the image to the desired size if a preproc function is provided
        if self.preproc is not None:
            img = self.preproc(img, self.img_size)

        return img, captions
