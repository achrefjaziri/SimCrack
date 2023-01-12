import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

"""
The data is available from https://www.robots.ox.ac.uk/~vgg/data/dtd/
"""


class DTDDataProvider:

    def __init__(self, save_path=None, train_batch_size=32, test_batch_size=200, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):

        norm_mean = [0.5329876098715876, 0.474260843249454, 0.42627281899380676]
        norm_std = [0.26549755708788914, 0.25473554309855373, 0.2631728035662832]

        valid_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        style_data = datasets.DatasetFolder(root='./dtd/images',)#datasets.ImageFolder(os.path.join(save_path, 'valid'), valid_transform)

        self.style_dataloder = torch.utils.data.DataLoader(
            style_data, batch_size=1, shuffle=True,
            pin_memory=True, num_workers=n_worker)

import torch
from torchvision import datasets, transforms

class DTD(datasets.ImageFolder):
    """Describable Textures Dataset (DTD)"""

    def __init__(self, root, transform=None):
        super(DTD, self).__init__(root, transform=transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a class label
        """
        img = self.data[index]

        # convert PIL image to tensor
        if self.transform is not None:
            img = self.transform(img)

        return img

import torch
from torchvision import datasets, transforms

class DatasetFolderWithoutLabels(datasets.DatasetFolder):
    """DatasetFolder without labels"""

    def __init__(self, root, loader, extensions, transform=None):
        super(DatasetFolderWithoutLabels, self).__init__(root, loader, extensions, transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            image: image data
        """
        path, _ = self.samples[index]
        sample = self.loader(path)

        # convert PIL image to tensor
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

def get_dtd_loader(batch_size, data_augmentation):
    """
    Load images from a directory without labels

    Args:
        batch_size (int): batch size
        data_augmentation (bool): If True, apply data augmentation

    Returns:
        DataLoader: Image data loader
    """

    # define transforms
    if data_augmentation:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
        ])

    # define image loader
    def default_image_loader(path):
        from PIL import Image
        return Image.open(path).convert('RGB')

    # load dataset
    dataset = DatasetFolderWithoutLabels(root='/data/resist_data/datasets/dtd/images',
                                         loader=default_image_loader,
                                         extensions=('.jpg', '.jpeg', '.png'),
                                         transform=transform)

    # create dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4)

    return dataloader





if __name__=='__main__':

    style_dataloder = get_dtd_loader(1,False)

    """
    style_data = datasets.DatasetFolder(
        root='/home/ajaziri/resist_projects/SimCrack/dtd/images',transform=loader)  # datasets.ImageFolder(os.path.join(save_path, 'valid'), valid_transform)
    style_dataloder = torch.utils.data.DataLoader(
            style_data, batch_size=1, shuffle=True,
            pin_memory=True, num_workers=16)

    """

    import matplotlib.pyplot as plt

    for idx, (img) in enumerate(style_dataloder):
        print(img.shape)

        img = img[0]
        print(img.shape)
        plt.Figure()

        plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()
