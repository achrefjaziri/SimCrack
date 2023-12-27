# from post_processing import *
import numpy as np
from PIL import Image
import glob as gl
import numpy as np
from PIL import Image
import torch
from sklearn.gaussian_process.kernels import RBF
import lap
import os
import csv
import cv2
from scipy.ndimage.morphology import binary_dilation
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

SMOOTH = 0.


def save_eval_history(header, value, file_name):
    """ export data to csv format
    Args:
        header (list): headers of the column
        value (list): values of correspoding column
        folder (list): folder path
        file_name: file name with path
    """
    # make new folder
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    # if file exists, add row else create new file
    if os.path.isfile(file_name) == False:
        file = open(file_name, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    else:
        file = open(file_name, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    file.close()


def iou_numpy(outputs: np.array, labels: np.array):
    """
    orignal code is taken from https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    :param outputs:
    :param labels:
    :return:
    """
    # outputs = outputs.squeeze(1)
    lab = labels.astype(bool)
    o = outputs.astype(bool)

    intersection = (o & lab).sum((1, 2))
    union = (o | lab).sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    lab = np.invert(lab)
    o = np.invert(o)

    intersection = (o & lab).sum((1, 2))
    union = (o | lab).sum((1, 2))
    iou = np.concatenate([(intersection + SMOOTH) / (union + SMOOTH), iou])

    # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return iou.mean()  # Or thresholded.mean()


def minWeightBipartitAcc(gtIndices, predIndices,length_scales=5):
    # gtIndices = np.argwhere(gt) #np array of shape (nb_of_white_pixesl,2)
    # predIndices = np.argwhere(prediction)  #np array of shape (nb_of_white_pixesl,2)
    if gtIndices.shape[0] == 0:
        return 1, 1
    elif predIndices.shape[0] == 0:
        # print("gt length ",gtIndices.shape[0])
        # print("pred length ",predIndices.shape[0])
        return 0, None

    rbf = RBF(length_scale=length_scales)  # imported from sklearn.gaussian_process.kernels
    costMatrix = rbf.__call__(gtIndices, predIndices)
    m2 = 1 - costMatrix
    try:
        cost, x, y = lap.lapjv(m2, extend_cost=True)
        cost = cost / (min(len(x), len(y)))
        min_pixels = min(gtIndices.shape[0], predIndices.shape[0])
        max_pixels = max(gtIndices.shape[0],predIndices.shape[0])

        per = min(1.0,(min_pixels+min_pixels * 0.2)/max_pixels) #10% tolerance

        #print(cost,per)

        return (1 - cost), per
    except MemoryError as err:
        print("Memory error. Cost matrix is too big")
        return 1, gtIndices.shape[0] / predIndices.shape[0]


def theta_F1(gt, pred, theta):
    dilated_gt = binary_dilation(gt, iterations=theta)
    pred = np.array(np.squeeze(pred), dtype=bool)
    y_true_f = dilated_gt.flatten().tolist()
    y_pred_f = pred.flatten().tolist()
    x = np.logical_and(y_true_f, y_pred_f)
    x = np.logical_and(y_true_f, not y_pred_f)
    print('gt and pred positives', np.sum(y_true_f), np.sum(y_pred_f))
    print(not y_true_f)

    tp = np.sum(np.logical_and(y_true_f, y_pred_f))
    fn = np.sum(np.logical_and(y_true_f, np.logical_not(y_pred_f)))
    fp = np.sum(np.logical_and(np.logical_not(y_true_f), y_pred_f))
    print(tp, fn, fp)
    score = tp / (tp + (1 / 2) * (fp + fn))

    return score
    # TODO not all f1 scores are the same. Please verify


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    tp = np.sum(np.logical_and(y_true_f, y_pred_f))
    fn = np.sum(np.logical_and(y_true_f, not y_pred_f))
    fp = np.sum(np.logical_and(not y_true_f, y_pred_f))
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (np.sum(y_true_f) + np.sum(y_pred_f) + SMOOTH)

    # i should try morphological delation


def get_neighbours(w, h, theta):
    "helper function for theta F1"
    neighbours = [[w + i, h + j] for j in range(-theta, theta) for i in range(-theta, theta)]
    return np.array(neighbours)


def theta_F1(ground_truth, prediction, theta):
    "computes the F1 score with tolerance theta"
    pred_indices = np.argwhere(prediction.flatten())
    gt_indices = np.argwhere(ground_truth.flatten())

    fn = np.setdiff1d(gt_indices, pred_indices)
    tp = np.intersect1d(gt_indices, pred_indices)
    fp = np.setdiff1d(pred_indices, gt_indices)

    w, h = prediction.shape[0], prediction.shape[1]
    gt_indices_non_flattened = np.argwhere(ground_truth)
    tolerated_mistakes = [any(n in gt_indices_non_flattened for n in get_neighbours(x // w, x % w, theta)) for x in
                          fp].count(True)
    # for x in fp:
    #    current_w = x // w
    #    current_h = x % w
    #    neighbours = get_neighbours(current_w, current_h, theta)
    #    if any(n in gt_indices_non_flattened for n in neighbours):
    #        tolerated_mistakes += 1

    f1_tolerance = (tp.shape[0] + tolerated_mistakes) / (
            tp.shape[0] + tolerated_mistakes + 0.5 * (fn.shape[0] + fp.shape[0] - tolerated_mistakes))
    return f1_tolerance


