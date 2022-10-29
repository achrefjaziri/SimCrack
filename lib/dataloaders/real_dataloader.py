"""
This dataloder loads an input image from the collected Real data.
The images are loaded from the subset of images that contain well-defined cracks. More challenging cases (cracks + other defects are currently out of scope)
"""

import numpy as np
from PIL import Image
from PIL import ImageOps
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from random import randint
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import copy
import random
import os
from PIL import ImageFile
from PIL import Image, ImageOps
from skimage import exposure


class RealDataloader(Dataset):
    """
    Basic Dataloader for images
    """
    def __init__(self, configs):
        super(RealDataloader, self).__init__()
        # all file names
        self.configs = configs
        # TODO The next line is hardcoded and should change to allow for different difficulties. clean_cracks, cracks_with_corrosion etc..
        path_dir = os.path.join(self.configs.data_input_dir,'resist_set/images', '*')
        self.image_arr = glob.glob(path_dir)
        print(len(self.image_arr))
        self.img_transforms = transforms.Compose([transforms.ToTensor()])
    def transform(self, image):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if self.configs.input_ch==1:
            input_image = np.asarray(image)
            input_image = input_image[:, :, :3]
            # greyscale input
            input_image = np.dot(input_image[..., :3], [0.299, 0.587, 0.114])
            input_image = torch.unsqueeze(torch.from_numpy(input_image),dim=0).float()
        else:
            input_image = self.img_transforms(image)
        return input_image
    def __getitem__(self, index):
        img_path = self.image_arr[index]
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image) #This is needed because some images are displayed in a different orientation than how they are saved. This can lead to issues in evaluation.
        if self.configs.arch_name == 'pmiunet':
            pmi_path = os.path.join(self.configs.pmi_dir,f'{self.configs.neighbour_size}_{self.configs.phi_value}'
                                    ,'RealResist',os.path.basename(img_path)+'.npy')
            image = np.load(pmi_path)
            image = image.transpose(2, 0, 1)
            if self.configs.input_ch==1:
                img = image[0]
                if self.configs.histequalize_pmi:
                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                    img = exposure.equalize_adapthist(img, clip_limit=0.03)
                image = np.expand_dims(img, axis=0)  # use only one PMI scale
            image = torch.tensor(image).float()
            return {'input': image, 'path': img_path}
        else:

            img = self.transform(image)
            return {'input': img, 'path': img_path}

    def __len__(self):
        return len(self.image_arr)



