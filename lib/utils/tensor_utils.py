import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import numpy as np
import math,os
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize



def get_padding_values(h,patch_size):
    n = h // patch_size
    if (h % patch_size) ==0:
        return 0,0
    else:
        additional_pixels = (n+1)* patch_size - h
        if (additional_pixels % 2) ==0:
            return additional_pixels//2,additional_pixels//2
        else:
            return (additional_pixels // 2)+1, (additional_pixels // 2)

def remove_padding(input_img,x1,y1,x2,y2):
    if x1 * y1 * x2 * y2 != 0:
        input_img = input_img[:, x1:-y1, x2:-y2]
    else:
        if x1 * y1 * x2 != 0:
            input_img = input_img[:, x1:y1, x2:-y2]
        elif x1 * y1 * y2 != 0:
            input_img = input_img[:, x1:-y1, :-y2]
        elif x1 * x2 * y2 != 0:
            input_img = input_img[:, x1:, x2:-y2]
        elif y1 * x2 * y2 != 0:
            input_img = input_img[:, :-y1, x2:-y2]
        elif x1 * y1 != 0:
            input_img = input_img[:, x1:-y1, :]
        elif x1 * y2 != 0:
            input_img = input_img[:, x1:, :-y2]
        elif x1 * x2 != 0:
            input_img = input_img[:, x1:, x2:]
        elif y1 * x2 != 0:
            input_img = input_img[:, :-y1, x2:]
        elif y1 * y2 != 0:
            input_img = input_img[:, :-y1, :-y2]
        elif x2 * y2 != 0:
            input_img = input_img[:, :, x2:-y2]
        elif x1 != 0:
            input_img = input_img[:, x1:, :]
        elif x2 != 0:
            input_img = input_img[:, :, x2:]
        elif y1 != 0:
            input_img = input_img[:, :-y1, :]
        elif y2 != 0:
            input_img = input_img[:, :, :-y2]
   #print(input_img.shape)
    return input_img