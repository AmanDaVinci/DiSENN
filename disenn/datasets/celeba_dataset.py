import os
import zipfile
import subprocess
import pandas as pd
import PIL
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from disenn.utils.download import download_file_from_google_drive


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

    IMG_DIR = "img_align_celeba/img_align_celeba/"
    ATTR_FILE = "list_attr_celeba.csv" 
    PARTITION_FILE = "list_eval_partition.csv"
    IMG_COL = 'image_id'
    PARTITION_MAP = {'train': 0, 'valid': 1, 'test': 3}
    IMG_CHANNELS = 3
    IMG_SIZE = 64
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    FILE_LIST = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

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
        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, data_path, filename, md5)
        with zipfile.ZipFile(data_path / "img_align_celeba.zip", "r") as f:
            f.extractall(data_path)
    
    def __len__(self):
        return self.df_images.shape[0]
    
    def __getitem__(self, index):
        img_name = self.df_images.iloc[index].name
        label = self.df_images.iloc[index][self.target]
        img = PIL.Image.open(self.data_path / self.IMG_DIR / img_name)
        img = self.transformations(img)
        return img, label