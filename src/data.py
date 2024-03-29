from PIL import Image
import glob
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class ImageTransform():
    """
    Image preprocessing class.
    Make transforms for train and val as a dictionary.
    Data augmentation is performed for training.

    Attributes
    ----------
    resize: int
        Image size after resized
    mean: (R, G, B)
        Mean value of each channel
    std: (R, G, B)
        Std value of each channel
    """
    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase: 'train' or 'val' or 'test'
            Specify the preprocessing mode.
        """
        return self.data_transform[phase](img)


def make_data_path_list(phase='train', rootpath='../data/'):
    """
    make list stored data path for train and test

    Parameters
    ----------
    phase: 'train' or 'test'
        Specify the preprocessing mode.

    Returns
    -------
    path_list: list
        list stored data path
    """
    target_path = osp.join(rootpath+phase+'/**.png')
    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)
        
    return path_list


class ImageDataset(Dataset):
    """
    Class for making dataset

    Attributes
    ----------
    file_list: list
        list stored data path
    train_label_master: dict
        dictionary of train picture and label
    transform: object
        instance for preprocessing
    phase: 'train' or 'val'
        Specify the preprocessing mode.
    """
    
    def __init__(self, file_list, train_label_master, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.train_label_master = train_label_master
        self.phase = phase

    def __len__(self):
        """
        Return number of pictures
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Get the tensor of preprocessed image and label
        """
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(
            img, self.phase)

        label = self.train_label_master[img_path[14:]]

        return img_transformed, label
