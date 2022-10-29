import argparse
import time
import argparse
import os
import torch
import numpy as np
from lib.arg_parser import train_args
from lib.arg_parser.eval_args import parse_eval_args
from lib.arg_parser.pmi_args import parse_pmi_args


def parse_args():
    # load parameters and options
    parser = argparse.ArgumentParser(description='PyTorch semantic segmentation training')

    parser = parse_general_args(parser)
    parser = train_args.parse_train_args(parser)
    parser = parse_pmi_args(parser)

    args = parser.parse_args()

    args.time = time.ctime()

    return args


def parse_args_eval():
    # load parameters and options
    parser = argparse.ArgumentParser(description='Model Evaluation')

    parser = parse_eval_args(parser)

    args = parser.parse_args()

    args.time = time.ctime()
    return args


def parse_general_args(parser):
    # General model parameters
    parser.add_argument('--arch_name', default='unet',
                        help='Possible Architecture: unet, munet, ganunet, pmiunet')

    parser.add_argument('--m', default='',
                        help='Custom message to add to the log file')

    parser.add_argument(
        "--test_mode",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Boolean to activate test mode otherwise we are in training mode",
    )

    parser.add_argument('--resume', default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Resume Training if we are in training mode')

    parser.add_argument("--num_epochs", default=30, type=int, help='Number of training epochs')

    parser.add_argument(
        "--device",
        default="cuda",
        help="cpu or cuda to use GPUs",
    )

    parser.add_argument(
        "--port",
        default="6010",
        help="Port specification needed for distributed multi-gpu training",
    )

    parser.add_argument(
        "--save_dir",
        default="./workspace",
        help="Create directory to save results in. Training results are saved in ./workspace/arcg_name/run_number",
    )

    # Parameters related to the dataset
    parser.add_argument(
        "--dataset",
        default='SimResist',
        help="Available Datasets: SimResist, RealResist, MultiSet",
    )
    parser.add_argument(
        "--partition_name",
        default='clean_cracks',
        help="This argument is only relevant for RealResist: crack_corrosion - clean_cracks - complicated",
    )
    parser.add_argument(
        "--data_input_dir",
        default='/data/resist_data/datasets/',
        help="Directory where the data is saved",
    )

    parser.add_argument(
        "--input_size",
        default=256,
        type=int,
        help=" Dimension of the input data",
    )

    parser.add_argument(
        "--resize_crop_input",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true the image will be resized to resize_size and then cropped to input_size"
             " otherwise the original image will be just downsized to input_size"
    )
    parser.add_argument(
        "--resize_size",
        default=512,
        type=int,
        help="Resize of the input image before cropping. This parameter is used when using simulated Data.",
    )

    parser.add_argument(
        "--input_ch",
        default=1,
        type=int,
        help="Number of input channels. This parameter does not matter for one channel datasets like MNIST",
    )
    parser.add_argument(
        "--num_classes",
        default=2,
        type=int,
        help="Number of classes for the training data (needed only for the supervised models CE and predSim) ",
    )

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size",
    )

    # Parameters related to multi GPU training
    parser.add_argument('--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('--gpus', default=2,
                        help='number of gpus per node')
    parser.add_argument('--world_size', default=None,
                        help='number of gpus per node')

    parser.add_argument('--nr', default=0,
                        help='ranking within the nodes')

    # Parameters to resume training and evaluation

    parser.add_argument(
        "--model_path",
        default='/home/ajaziri/resist_projects/SimCrack/workspace/unet/SimResist/2022-09-22_09-39-19',
        help="Path to the directory containing the model",
    )


    parser.add_argument(
        "--patchwise_eval",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Evaluate Real Images Patchwise",
    )

    parser.add_argument(
        "--patch_size",
        default=512,
        type=int,
        help="Size of the  cropped patch before downsizing to input_size. This parameter is used when doing patchwise evaluation in run_segmentation.py script.",
    )


    # Parameters related to visualization of validation results

    parser.add_argument(
        "--save_val_examples",
        default=False,
        help="Save Validation examples in a folder each epoch",
    )
    parser.add_argument(
        "--vis_freq",
        default=50,
        type=int,
        help="Determines how often validation examples are saved",
    )
    return parser
