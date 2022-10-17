""" Custom functions for calculating the performance of the models used
in the crackproject.

DepthLoss = calculate losses for depth segmentation
intersection_over_union = calculate the IOU for semantic segmentation

sources:
DepthLoss = https://github.com/DhruvJawalkar/Depth-Map-Prediction-from-a-Single-Image-using-a-Multi-Scale-Deep-Network/blob/master/model_utils.py
"""

import torch
import torch.nn.functional as F

class DepthLoss(torch.nn.Module):
  """Loss function for depth segmentation"""
  def __init__(self):
    super(DepthLoss, self).__init__()

  def im_gradient_loss(self, d_batch, n_pixels):
    """Calculate the gradient loss"""
    device = torch.device('cuda')
    # define kernels for edge detection
    kernel_a = torch.Tensor([[[[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]]]])

    kernel_b = torch.Tensor([[[[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]]]])
    # put the kernels on cuda device
    a = kernel_a.to(device)
    b = kernel_b.to(device)

    conv_x = F.conv2d(d_batch, a, padding=1).to(device)
    conv_y = F.conv2d(d_batch, b, padding=1).to(device)
    conv_sum = torch.pow(conv_x, 2) + torch.pow(conv_y, 2)
    return  conv_sum.view(-1, n_pixels).mean(dim=1).sum()

  def forward(self, outputs, targets):
    """calculate the final depth loss term"""

    # access the width and height of the target
    n_pixels = targets.shape[2] * targets.shape[3]

    #outputs = (outputs * 0.225) + 0.45
    #outputs = outputs * 255
    #targets = (targets * 0.225) + 0.45
    #targets = targets * 255
    # assign infinitesimal small values to outputs that are equal to zero
    outputs[outputs <= 0] = 0.00001
    targets[targets <= 0] = 0.00001

    # after preprocessing calculate the difference of the logarithms between the outputs and the targets
    # torch log returns the natural logarithm of the input
    d = (torch.log(outputs) - torch.log(targets))

    # calculate the loss terms
    grad_loss_term = self.im_gradient_loss(d, n_pixels)
    term_1 = torch.pow(d.view(-1, n_pixels), 2).mean(dim=1).sum()
    term_2 = (torch.pow(d.view(-1, n_pixels).sum(dim=1), 2) / (2 * (n_pixels ** 2))).sum()
    return term_1 - term_2  #+ grad_loss_term



