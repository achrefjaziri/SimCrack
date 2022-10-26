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
from skimage import exposure
import matplotlib.pyplot as plt


if __name__=="__main__":
    npy_path = '/data/resist_data/pmi_maps/MultiSet/5_1.75/test/Rissbilder_for_Florian_9S6A2869_37_1430_3692_3865.jpg.npy'
    pmi_maps = np.load(npy_path)
    print(pmi_maps.shape)

    # Equalization
    img_eq = exposure.equalize_hist(pmi_maps[:,:,0])

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(pmi_maps[:,:,0], clip_limit=0.03)

    plt.figure()
    plt.imshow(pmi_maps[:,:,0])
    plt.show()

    plt.figure()
    plt.imshow(img_eq)
    plt.show()

    plt.figure()
    plt.imshow(img_adapteq)
    plt.show()






