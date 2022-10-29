"""
The goal of these functions is to compute the accuracy scores between predicted segmentation maps and ground truths using a variety of metrics.
The results for each prediction are saved in a csv file indicating the model id and the evaluation dataset.
"""

from PIL import Image
from lib.utils.eval_metrics import minWeightBipartitAcc,theta_F1
from lib.utils.save_history import save_eval_history
from lib.utils.parallel_hausdorff.hausdorff import smooth_hausdorff
import pandas as pd
import numpy as np
import math, os
import cv2
from torchvision import transforms
from sklearn.metrics import f1_score
from lib.arg_parser.general_args import parse_args_eval



def compute_scores(input_args):
    """
    compute accuracy scores and adds  them to csv file
    :param paths: paths[0] the path of the ground truth and paths[1] the path of the predictions
    :param csv_file: csv file to save the results
    :param args: to deterimne which metrics to use during the evaluation
    :return:
    """
    args, csv_file,paths  = input_args[0],input_args[1],input_args[2]

    prediction = cv2.imread(paths[1], cv2.IMREAD_GRAYSCALE)
    prediction =  cv2.resize(prediction, (256,256), interpolation = cv2.INTER_AREA)
    thresh = 128
    prediction = cv2.threshold(prediction, thresh, 255, cv2.THRESH_BINARY)[1]
    prediction = np.expand_dims(prediction, axis=0)
    prediction = prediction/255
    prediction = prediction.astype(int)

    #remove padding artifacts
    prediction[:,0]=0
    prediction[:,-1]=0


    # print(mask_name)
    ground_truth = Image.open(paths[0])
    #print('ground truth shape',np.array(ground_truth).shape)
    mask_transforms = transforms.Compose([transforms.Resize([prediction.shape[1], prediction.shape[2]])])
    ground_truth = mask_transforms(ground_truth)
    ground_truth = np.array(ground_truth, dtype=np.int)
    ground_truth = np.where(ground_truth > 0, 1, ground_truth)
    # delete separate color channel
    if args.dataset=='RealResist' or args.dataset=='SimResist':
        ground_truth = ground_truth[:, :, 0]




    #acc_val = iou_numpy(prediction, ground_truth)
    #print('final shapes,',ground_truth.shape,prediction.shape)
    acc_val = f1_score(ground_truth.ravel(),prediction.ravel(),average='binary')

    gtIndices = np.argwhere(ground_truth)  # np array of shape (nb_of_white_pixesl,2)
    predIndices = np.argwhere(np.squeeze(prediction))  # np array of shape (nb_of_white_pixesl,2)


    # Skip the intensive computation for WCN and Hausdorff metrics if the ground truth contains only the background class.
    if gtIndices.shape[0] == 0:
        print('score:', acc_val)
        wcn, per = None, None
        s_hausdorff_euc = None
        s_hausdorff_rbf = None
    else:
        # Returns the WCN scores and predIndices.shape[0]/gtIndices.shape[0] score
        if args.wcn:
            wcn,per = minWeightBipartitAcc(gtIndices,predIndices)
        else:
            wcn,per = None, None

        # if the predictions is empty while the ground truth contains a crack, we skip the computation for the hausdorff metric
        if args.hausdorff_rbf:
            if predIndices.shape[0]==0:
                s_hausdorff_rbf,s_hausdorff_euc =1.0,None
            else:
                s_hausdorff_rbf = smooth_hausdorff(gtIndices, predIndices, distance='rbf')
        else:
            s_hausdorff_rbf = None

        if args.hausdorff_euc:
            if predIndices.shape[0] == 0:
                s_hausdorff_euc = 1.0 #TODO check if this is correct?
            else:
                s_hausdorff_euc = smooth_hausdorff(gtIndices, predIndices, distance='euclidean')
        else:
            s_hausdorff_euc = None

        if args.f1_theta:
            if predIndices.shape[0] == 0:
                f1_2,f1_5 = 0.0,0.0
            else:
                # TODO check if this is correct?
                f1_2 = theta_F1(ground_truth, prediction, 10)
                f1_5 = theta_F1(ground_truth, prediction, 5)
        else:
            f1_2,f1_5 = None, None

        # Creating Header
        d = {'Img Name': [], 'F1': [], 'WCN': [], 'WCN_PER': [],
             'Hausdorff_EUC': [],
             'Hausdorff_RBF': [], 'F1_Theta2': [], 'F1_Theta5': []}

        df = pd.DataFrame(data=d)

        new_row = {'Img Name': os.path.basename(paths[1]), 'F1': acc_val,
                   'WCN': wcn, 'WCN_PER': per,
                   'Hausdorff_EUC':s_hausdorff_euc,
                   'Hausdorff_RBF': s_hausdorff_rbf,
                   'F1_Theta2': f1_2, 'Acc1': f1_5}

        save_eval_history(df, new_row, csv_file)
        print(f'Evaluation for {os.path.basename(paths[1])} is done.')


if __name__=="__main__":
    args_eval = parse_args_eval()

    csv_file_test = 'testing_file.csv'
    #paths for testing
    path_examples =['','']

    compute_scores(paths=path_examples,csv_file=csv_file_test,args=args_eval)

