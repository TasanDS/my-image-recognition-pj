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
    target_path = osp.join(rootpath+phase+'/**.png')
    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)
        
    return path_list


class ImageDataset(Dataset):
    def __init__(self, file_list, train_label_master, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.train_label_master = train_label_master
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(
            img, self.phase)

        label = self.train_label_master[self.train_label_master['file_name']==img_path[14:]]['label_id'].item()

        return img_transformed, label
