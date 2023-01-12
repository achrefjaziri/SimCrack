# -*- coding:utf-8 -*-
"""
this file implements a smooth hausdorff evaluation metric
"""
import os
import numpy as np
#from evaluation.hausdorff import distances
from inspect import getmembers
from scipy.spatial.distance import cdist
import time
from sklearn.gaussian_process.kernels import RBF

def rbf(l1,l2):
    #c = cdist(l1,l2)
    rbf = RBF(length_scale=5)  # imported from sklearn.gaussian_process.kernels
    costMatrix = rbf.__call__(l1, l2)
    m2 = 1 - costMatrix
    #print(c)
    return np.min(m2,axis=1)
def cdis(l1,l2):
    c = cdist(l1,l2)
    return np.min(c,axis=1)
def smooth_hausdorff(ground_truth, prediction, distance='euclidean'):
    """
    computes related hausdorff distance between two sets, `ground_truth` and
    `prediction`. Both sets don't have to have the same cardinality, but each
    element has to have the same dimension across both sets.

    param: ground_truth: np array of size [gt_samples, dim]
    param: prediction: np array of size [pred_samples, dim]

    """
    return hausdorff_distance(ground_truth, prediction,
                              distance=distance, mode='mean')


##############################################################################
### code base by mavillian (GitHub)
###############################################################################

def _hausdorff(XA, XB, distance, mode='max', k=6):
    nA = XA.shape[0]
    nB = XB.shape[0]
    cmax = 0
    cmins = []
    csum = 0
    if distance =='euclidean':
        cmins = cdis(XA,XB)
    else:
        cmins = rbf(XA,XB)



    if mode == 'mean':
        cmax = np.mean(cmins)
    elif mode == 'kmax':
        cmax = np.sort(cmins)[k]

    if distance == 'euclidean':
        cmins = cdis(XB, XA)
    else:
        cmins = rbf(XB, XA)

    if mode == 'mean':
        mean = np.mean(cmins)
        cmax = cmax if mean < cmax else mean

    elif mode == 'kmax':
        kmax = np.sort(cmins)[k]
        cmax = cmax if kmax < cmax else kmax
    return cmax

def _hausdorff2(XA, XB, distance_function, mode='max', k=6):
    nA = XA.shape[0]
    nB = XB.shape[0]
    cmax = 0
    cmins = []
    csum = 0
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = distance_function(XA[i,:], XB[j,:])
            if d<cmin:
                cmin = d
            if cmin<cmax and mode == 'max':
                break

        if mode != 'max' and np.inf > cmin:
            cmins.append(cmin)
            csum += cmin
        if mode == 'max' and cmin>cmax and np.inf>cmin:
            cmax = cmin

    if mode == 'mean':
        cmax = csum / nA
    elif mode == 'kmax':
        cmax = np.sort(cmins)[k]


    cmins = []
    for j in range(nB):
        cmin = np.inf
        for i in range(nA):
            d = distance_function(XA[i,:], XB[j,:])
            if d<cmin:
                cmin = d
            if mode == 'max' and cmin<cmax:
                break

        if mode != 'max' and np.inf > cmin:
            cmins.append(cmin)
        if mode == 'max' and cmin>cmax and np.inf>cmin:
            cmax = cmin

    if mode == 'mean':
        mean = np.mean(cmins)
        cmax = cmax if mean < cmax else mean

    elif mode == 'kmax':
        kmax = np.sort(cmins)[k]
        cmax = cmax if kmax < cmax else kmax
    return cmax

def _find_available_functions(module_name):
	all_members = getmembers(module_name)
	available_functions = [member[0] for member in all_members]
	return available_functions

def hausdorff_distance(XA, XB, distance='euclidean', mode='mean'):

    assert type(XA) is np.ndarray and type(XB) is np.ndarray, \
        'arrays must be of type numpy.ndarray'
    assert np.issubdtype(XA.dtype, np.number) and \
        np.issubdtype(XA.dtype, np.number), \
            'the arrays data type must be numeric'
    assert XA.ndim == 2 and XB.ndim == 2, 'arrays must be 2-dimensional'
    assert XA.shape[1] == XB.shape[1], \
        'arrays must have equal number of columns'
    assert mode in ['max', 'kmax', 'mean'], 'unknown hausdorff smoother'

    #if isinstance(distance, str):
    #    assert distance in _find_available_functions(distances), \
    #        'distance is not an implemented function'
    #    if distance == 'haversine':
    #        assert XA.shape[1] >= 2, \
    #        'haversine distance requires at least 2 coordinates per point (lat, lng)'
    #        assert XB.shape[1] >= 2, \
    #        'haversine distance requires at least 2 coordinates per point (lat, lng)'
    #     distance_function = getattr(distances, distance)
    #elif callable(distance):
    #    distance_function = distance
    #else:
    #    raise ValueError("Invalid input value for 'distance' parameter.")
    return _hausdorff(XA, XB, distance, mode=mode)




def main():
    """
    calls small demo example for hausdorff distance
    """
    def input_arrays():
        np.random.seed(42)
        XA = np.random.random((1000,2))
        XB = np.random.random((2000,2))
        return (XA,XB)

    print(smooth_hausdorff(*input_arrays()))



if __name__ == '__main__':
    #gt = np.argwhere(np.array([0,0,0]))
    #print(gt)
    #main()
    #np.random.seed(42)
    #XA = np.random.random((2, 2))
    XA =np.array([[1,1],[2,2]])
    #XB = np.random.random((3, 2))
    XB =np.array([[1,1],[2,2],[4,4]])
    rbf = RBF()  # imported from sklearn.gaussian_process.kernels
    costMatrix = rbf.__call__(XA, XB)
    m2 = 1 - costMatrix
    print(m2.shape)
    print(costMatrix)
    #print(cdist(XA, XB))
    #print(c(XA, XB))