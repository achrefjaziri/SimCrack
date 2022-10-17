import numpy as np
from datetime import datetime
import os, logging
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from lib.models.unet import UNet
from lib.dataloaders.sim_dataloader import SimDataloader
from lib.training.train import train_model
from lib.training.validate import validate_model
from lib.utils.save_history import save_models
from lib.utils.custom_depth_loss import DepthLoss
from lib.arg_parser.general_args import parse_args,parse_args_eval
from skimage import exposure
import matplotlib.pyplot as plt


if __name__=="__main__":
    npy_path = '/data/resist_data/pmi_maps/5_1.75/RealResist/strymonas_0000162_3000_2532.png.npy'
    pmi_maps = np.load(npy_path)
    print(pmi_maps.shape)

    # Equalization
    img_eq = exposure.equalize_hist(pmi_maps[:,:,1])

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(pmi_maps[:,:,1], clip_limit=0.03)

    plt.figure()
    plt.imshow(pmi_maps[:,:,1])
    plt.show()

    plt.figure()
    plt.imshow(img_eq)
    plt.show()

    plt.figure()
    plt.imshow(img_adapteq)
    plt.show()






