import os
import subprocess
import pandas as pd
import PIL
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebA(Dataset):
    """Large-scale CelebFaces Attributes (CelebA) Dataset [1]
    
    Returns the face image and the corresponding "male" attribute as 0 or 1

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes 
    dataset with more than 200K celebrity images, each with 40 attribute annotations. 
    The images in this dataset cover large pose variations and background clutter. 
    CelebA has large diversities, large quantities, and rich annotations, including
    - 10,177 number of identities,
    - 202,599 number of face images, and
    - 5 landmark locations, 40 binary attributes annotations per image.
    The dataset can be employed as the training and test sets for the following 
    computer vision tasks: face attribute recognition, face detection, landmark 
    (or facial part) localization, and face editing & synthesis. 
    
    NOTE
    ----
    The dataset will be downloaded from kaggle

    References
    ----------
     [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face 
     attributes in the wild. In Proceedings of the IEEE international conference 
     on computer vision (pp. 3730-3738).
    """

    KAGGLE_FILE = "jessicali9530/celeba-dataset"
    IMG_DIR = "img_align_celeba/img_align_celeba/"
    ATTR_FILE = "list_attr_celeba.csv" 
    PARTITION_FILE = "list_eval_partition.csv"
    IMG_COL = 'image_id'
    PARTITION_MAP = {'train': 0, 'valid': 1, 'test': 3}
    IMG_CHANNELS = 3
    IMG_SIZE = 64

    def __init__(self, split: str, data_path: str, download: bool = False, target: str = 'Male'):
        """ Initialize the dataset configurations

        Parameters
        ----------
        split: str
            'train', 'valid' or 'test'

        data_path : str
            the path to store the data     
        """
        super().__init__()
        
        data_path = Path(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        if download: self._download(data_path)
        
        df_attributes = pd.read_csv(data_path / self.ATTR_FILE, index_col = self.IMG_COL)
        df_attributes.replace(to_replace=-1, value=0, inplace=True)
        df_partitions = pd.read_csv(data_path / self.PARTITION_FILE, index_col= self.IMG_COL)
        df_partitions = df_partitions.join(df_attributes[target], how='inner')
        self.df_images = df_partitions[df_partitions['partition'] == self.PARTITION_MAP[split]]
        
        self.transformations = transforms.Compose([
            transforms.Resize(self.IMG_SIZE),
            transforms.CenterCrop(self.IMG_SIZE),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

        self.data_path = data_path
        self.target = target

    def _download(self, data_path):
        """Downloads the celeba dataset using kaggle api"""
        subprocess.check_call(["kaggle", "datasets", "download",
                                self.KAGGLE_FILE, "--path", data_path, "--unzip"])
    
    def __len__(self):
        return self.df_images.shape[0]
    
    def __getitem__(self, index):
        img_name = self.df_images.iloc[index].name
        label = self.df_images.iloc[index][self.target]
        img = PIL.Image.open(self.data_path / self.IMG_DIR / img_name)
        img = self.transformations(img)
        return img, label