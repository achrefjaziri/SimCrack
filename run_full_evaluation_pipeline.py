"""
This script evaluates all networks found under /workspace/trained_nets.
evalaution steps for each network:
1- run_segmentatiopn.py: creates segmentation maps for all images in test set
2- run_parallel_eval.py: computes the  accuracy scores for each image and saves them in csv files
"""
from run_parallel_eval import evaluate
from lib.arg_parser.general_args import parse_args, parse_args_eval

import logging
import os
import glob
import subprocess
import torch
from torch.utils.data import DataLoader
from run_segmentation import predict_cracks


def str_to_bool(s):
    #print('str',s)
    if 'True' in s:
         return True
    elif 'False' in s:
         return False
def get_log_value(f, class_name, type='string'):
    '''
    Extracts parameter value from .log file
    :param f: the lines of the log file
    :param class_name:  the class that we are interested in
    :param type:  type of output. Default is string.
    :return:  returns value of the class_name found in f file.
    '''
    matching = [s for s in f if class_name in s]
    if matching == []:
        if type == 'boolean':
            return False
        elif type == 'int':
            return 0
        elif type == 'float':
            return 0
        else:
            return ''

    my_string = matching[-1]
    if type == 'boolean':
        #print('newwww')
        out_string= my_string.split(class_name, 1)[1]
        out_string = out_string.replace("\n", "")
        #print(my_string.split(class_name, 1)[1], str_to_bool('False'), str_to_bool(out_string))
        return str_to_bool(out_string)
    elif type == 'int':
        return int(my_string.split(class_name, 1)[1])
    elif type == 'float':
        return float(my_string.split(class_name, 1)[1])
    else:
        out_string = my_string.split(class_name, 1)[1]
        out_string = out_string.replace(" ", "")
        out_string = out_string.replace("\n", "")
        return out_string


def get_info(model_path):
    """
    :param dict_res: dictionary containing all the current results
    :return: a dictionary containing the following model information : training_set, batch_size,lr, patch_size, phi_value, hist_eq
    """

    train_log = os.path.join(model_path,
                             'run_history_0.log')  # contains info about training batch size, lr , phi_values etc.

    with open(train_log) as f:
        f = f.readlines()

    phi_value = get_log_value(f, 'PMI Phi Value:', type='float')
    hist_eq = get_log_value(f, 'PMI Hist Eq:', type='boolean')
    input_ch = get_log_value(f, 'Input Channels:', type='int')
    arch_name = get_log_value(f, 'Architecture:', type='string')
    train_dataset = get_log_value(f, 'Dataset:', type='string')
    set_size = get_log_value(f, 'Set Size:', type='string')

    fuse_pred = get_log_value(f, 'Fuse Predictions:', type='boolean')
    att_connections = get_log_value(f, 'Attention Connections:', type='boolean')
    cons_loss = get_log_value(f, 'Consistency Loss:', type='boolean')
    #print('CONS LOSS',cons_loss)
    if train_dataset=='':
        train_dataset='SimResist'
    if input_ch==0:
        input_ch=True

    return arch_name, train_dataset,set_size, phi_value, hist_eq, input_ch, fuse_pred, att_connections, cons_loss


if __name__ == "__main__":

    args = parse_args_eval()
    all_trained_networks = glob.glob(os.path.join(args.save_dir, 'trained_nets', '*', '*', '*')) #2023-02-01_07-58-26

    # sanity check to remove all empty directories
    non_empty_dir = []
    for path in all_trained_networks:
        if os.path.exists(os.path.join(path, 'best_model.pth.tar')):
            non_empty_dir.append(path)

    all_trained_networks = non_empty_dir


    for model_path in all_trained_networks:
        print('current Model....', model_path)
        arch_name, train_dataset, set_size, phi_value, hist_eq, input_ch, fuse_pred, att_connections,cons_loss = get_info(model_path)

        args_inference = parse_args()
        args_inference.test_mode = True
        args_inference.save_dir = args.save_dir
        args_inference.arch_name = arch_name
        args_inference.model_path = model_path
        args_inference.input_ch = input_ch
        args_inference.phi_value = phi_value
        args_inference.histequalize_pmi = hist_eq
        args_inference.input_size = args.input_size
        args_inference.patchwise_eval = args.patchwise_eval
        args_inference.dataset = args.dataset
        args_inference.fuse_predictions = fuse_pred
        args_inference.att_connection = att_connections
        args_inference.cons_loss = cons_loss
        print(fuse_pred,att_connections,cons_loss)
        predict_cracks(args_inference)

        pred_path = os.path.join(args.save_dir, "eval_outputs", args.dataset, arch_name, os.path.basename(model_path))

        args_eval = parse_args_eval()
        args_eval.pred_path = pred_path
        args_eval.train_dataset = train_dataset
        args_eval.set_size = set_size

        evaluate(args_eval)

        '''
        print('COMMAND',[python_exec, '/home/ajaziri/resist_projects/SimCrack/run_segmentation.py', '--test_mode',
                        '--save_dir', args.save_dir, '--arch_name', arch_name, '--model_path',model_path, '--input_ch', str(input_ch),
                        '--phi_value', str(phi_value),histequalize_pmi,'--input_size', str(args.input_size), patchwise_eval, '--dataset',args.dataset
                        ])

        subprocess.run([python_exec, '/home/ajaziri/resist_projects/SimCrack/run_segmentation.py', '--test_mode',
                        '--save_dir', args.save_dir, '--arch_name', arch_name, '--model_path',model_path, '--input_ch', str(input_ch),
                        '--phi_value', str(phi_value),histequalize_pmi,'--input_size', str(args.input_size), patchwise_eval, '--dataset',args.dataset
                        ])


        subprocess.run([python_exec, '/home/ajaziri/resist_projects/SimCrack/run_parallel_eval.py',
                        '--save_dir', args.save_dir, '--gt_path', args.gt_path, '--pred_path', pred_path, '--rbf_l',
                        str(args.rbf_l),
                        '--train_dataset', train_dataset,
                        '--dataset', args.dataset
                        ])
        '''
    print('all Eval Done..')

# args
# get name of all networks and their ids + parameters
# check if they were already evaluated if yes remove from the list
# do the run_segmentation and run_parallel_eval.py


# Finish this script and start full on training.
# Finish 3 Papers to the summary stack
# Read One paper thorouly
# Recap the functional analysis stuff
