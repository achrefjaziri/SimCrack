"""
This Dataloder loads an input image and its corresponding segmentation mask.
The images are loaded from  an input directory which contains two directories /inputs and /gsround_truths.
This dataloader expects that the all images in the input folder have a ground truth map with the same naming convention.
"""
import numpy as np
from PIL import Image
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
import OpenEXR
import array
import Imath
import cv2
import os
import random
import matplotlib.pyplot as plt
from PIL import ImageFile
from lib.arg_parser.general_args import parse_args
from skimage import exposure
import kornia


class MixedDataloader(Dataset):
    # Class for loading data generated by the simulator for training or testing.
    def __init__(self, configs, mode='train', set_size=''):
        super(MixedDataloader, self).__init__()

        # all file names
        self.configs = configs
        imgs_path_render = os.path.join(self.configs.data_input_dir, 'sim_crack', mode, "images", '*')
        imgs_path_real = os.path.join(self.configs.data_input_dir, 'crack_segmentation_dataset', mode, "images", '*')

        self.image_arr = glob.glob(imgs_path_render) + glob.glob(imgs_path_real)

        self.img_size = configs.input_size
        self.mode = mode

        # Calculate len
        self.data_len = len(self.image_arr)
        if set_size != '':
            self.image_arr.sort()
            self.image_arr = self.image_arr[:int(set_size)]
        print(f'data length for {mode}', self.data_len)

        if self.configs.resize_crop_input:
            # transformations of the image tensor
            self.scale_size = self.configs.resize_size
            self.img_transforms = transforms.Compose([transforms.Resize([self.scale_size, self.scale_size]),
                                                      # transforms.CenterCrop(self.in_size),transforms.ToTensor()
                                                      ])

            # transformations of the segmentation tensor
            self.seg_transforms = transforms.Compose([transforms.Resize([self.scale_size, self.scale_size]),
                                                      # transforms.CenterCrop(self.out_size)
                                                      ])

        else:

            self.img_transforms = transforms.Compose([transforms.Resize([self.img_size, self.img_size]),
                                                      ])

            # transformations for the mask tensor
            self.seg_transforms = transforms.Compose([transforms.Resize([self.img_size, self.img_size]),
                                                      ])

    def transform_crop_downsize_combo(self, image, segmask,rendered_img):
        '''
        Transforms inputs and ground truths by first downsizing to self.scale_size and then randomly cropping to self.input_size
        :param image:
        :param segmask:
        :param depth:
        :param normal:
        :param pmi_gt
        :return:
        '''

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = self.img_transforms(image)
        mask = self.seg_transforms(segmask)


        """Image Cropping"""
        if rendered_img:
            mask_cropped = np.zeros((self.img_size, self.img_size, 4))
            count = 0
            while (np.count_nonzero(mask_cropped[:, :, 0]) < 200) and (
                    count < 50):  # makes sure that the training image contains a crack in most cases
                count = count + 1
                i, j, h, w = transforms.RandomCrop.get_params(
                    mask, output_size=(self.img_size, self.img_size))

                if self.configs.arch_name != 'pmiunet':
                    image_cropped = transforms.functional.crop(image, i, j, h, w)
                    image_cropped = np.asarray(image_cropped)

                else:
                    image_cropped = image[i:i + h, j:j + w, :]



                mask_cropped = transforms.functional.crop(mask, i, j, h, w)
                mask_cropped = np.asarray(mask_cropped)
        else:
            mask_cropped = np.zeros((self.img_size, self.img_size, 3))
            count = 0
            while (np.count_nonzero(mask_cropped[:, :]) < 200) and (
                    count < 50):  # makes sure that the training image contains a crack in most cases
                count = count + 1
                i, j, h, w = transforms.RandomCrop.get_params(
                    mask, output_size=(self.img_size, self.img_size))

                if self.configs.arch_name != 'pmiunet':
                    image_cropped = transforms.functional.crop(image, i, j, h, w)
                    image_cropped = np.asarray(image_cropped)

                else:
                    image_cropped = image[i:i + h, j:j + w, :]

                mask_cropped = transforms.functional.crop(mask, i, j, h, w)
                mask_cropped = np.asarray(mask_cropped)

        # delete alpha channel
        input_image = image_cropped[:, :, :3]
        if self.configs.input_ch == 1 and self.configs.arch_name != 'pmiunet':
            # greyscale input
            input_image = np.dot(input_image[..., :3], [0.299, 0.587, 0.114])
            input_image = np.expand_dims(input_image, axis=0)
            input_torch = torch.tensor(input_image).float()
        else:
            input_image = input_image.transpose(2, 0, 1)  # Permute axis to obtain image with shape (c,h,w)
            input_torch = torch.tensor(input_image).float()

        mask_torch = torch.from_numpy(np.array(mask_cropped, dtype=int))
        # assing integer values to the pixels
        mask_torch = np.where(mask_torch > 0, 1, mask_torch)
        if rendered_img:
            # delete separate color channel for rendered imgs
            mask_torch = mask_torch[:, :, 0]
        return input_torch, mask_torch

    def transform(self, image, mask, rendered_img=False):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        """Image Cropping"""
        image = self.img_transforms(image)
        mask = self.seg_transforms(mask)
        # delete alpha channel
        image = np.asarray(image)  # .transpose(1,2,0)
        image = image[:, :, :3]

        # TODO change this to add pmi_gt
        if self.configs.input_ch == 1:
            # greyscale input
            # image = np.asarray(image).transpose(1,2,0)
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            image = np.expand_dims(image, axis=0)
        else:
            image = image.transpose(2, 0, 1)  # Permute axis to obtain image with shape (c,h,w)

        mask = torch.from_numpy(np.array(mask, dtype=int))
        # assing integer values to the pixels
        mask = np.where(mask > 0, 1, mask)
        # delete separate color channel in case of rendered images
        if rendered_img:
            mask = mask[:, :, 0]
        image = torch.tensor(image).float()

        return image, mask

    def __getitem__(self, index):
        img_path = self.image_arr[index]
        rendered_img= 'render' in img_path
        if rendered_img: #All images from Cracktal contain render in them
            mask_path = img_path.replace("images", "gts").replace('render',
                                                              'gt')  # This probably needs to be changed depending on our structure
        else: #multi source set images
            mask_path = img_path.replace("images", "masks")  # This probably needs to be changed depending on our structure


        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.configs.resize_crop_input:
            img, gt = self.transform_crop_downsize_combo(image, mask,rendered_img)
        else:
            img, gt = self.transform(image, mask,rendered_img)


        data = {'input': img,
                'gt': gt,
                'path': mask_path}

        return data

    def __len__(self):
        return len(self.image_arr)


if __name__ == "__main__":
    args = parse_args()
    Dataset_test = MixedDataloader(args, mode='train')

    test_load = \
        torch.utils.data.DataLoader(dataset=Dataset_test,
                                    num_workers=2, batch_size=1, shuffle=True)
    for batch, data in enumerate(test_load):
        input_img = data['input'].detach().cpu().numpy()
        gt = data['gt'][0].detach().cpu().numpy()

        print('image', input_img.shape, gt.shape)

        plt.figure()
        plt.imshow(input_img[0,:, :, :].transpose(1, 2, 0))
        plt.show()
        # plt.figure()
        # plt.imshow(target_img)
        # plt.show()

        # plt.Figure()
        # plt.imshow(input_img,cmap='gray')
        # plt.savefig(f"./sim_examples/input{batch}.png")

        # plt.Figure()
        # plt.imshow(target_img,cmap='gray')
        # plt.savefig(f"./sim_examples/gt{batch}.png")