import argparse




def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
def parse_common_evaluation_args(parser):
    parser.add_argument(
        "--save_dir",
        default="/data/resist_data/SimCrack/workspace",
        help="Create directory to save results.",
    )

    parser.add_argument('--dataset', default='SimResist',
                        help='Current Dataset for Evaluation. '
                             'This argument is needed because each Dataset has a different naming scheme.'
                             'Available Datasets: SimResist, RealResist, CrackForest')
    parser.add_argument(
        "--patchwise_eval",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Evaluate Real Images Patchwise",
    )

    parser.add_argument(
        "--input_size",
        default=256,
        type=int,
        help=" Dimension of the input data",
    )

    parser.add_argument(
        "--set_size",
        default='',
        help="Set size used for training",
    )

    return parser


def parse_evaluation_args(parser):
    # Defines the parameters for run_full_evaluation_pipeline.py. This script combines both the run_segmentation and run_parallel evaluation.
    parser.add_argument('--wcn', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='include Weighted Closest Neighbour')
    parser.add_argument('--hausdorff_rbf', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='include Hausdorff with RBF Kernel')
    parser.add_argument('--hausdorff_euc', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='include Hausdorff with Euclidean Distance Kernel')
    parser.add_argument('--f1_theta', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='include F1 with tolerance')

    # Location of ground truth and prediction maps
    parser.add_argument('--gt_path', default='/data/resist_data/datasets/sim_crack/val/gts',
                        help='folder containing ground truth maps')  # /data/resist_data/datasets/resist_set/gts ,/data/resist_data/datasets/crack_segmentation_dataset/test/masks
    parser.add_argument('--pred_path',
                        default='/home/ajaziri/resist_projects/SimCrack/workspace/eval_outputs/SimResist/unet/2022-09-22_09-39-19',
                        help='folder containing prediction maps. The basename of this folder will be used to name csv files')


    parser.add_argument('--train_dataset', default='SimResist',
                        help='Dataset used to train the network. '
                             'This argument is needed because we want to get some arguments from training log files')

    parser.add_argument('--rbf_l', type=int, default=5,
                        help='l for rbf kernel')

    parser.add_argument(
        "--num_cpus",
        default=32,
        type=int,
        help="Number of cpu cores used for parallel evaluation",
    )

    parser.add_argument('--data_root_path', type=str, default='../../DATASETS/datasets_seg/GTA5',
                            help="the root path of dataset")
    parser.add_argument('--list_path', type=str, default='../datasets/GTA_640/list',
                            help="the root path of dataset")
    parser.add_argument('--checkpoint_dir', default="./log/gta5_pretrain_2",
                            help="the path of ckpt file")
    parser.add_argument('--xuanran_path', default=None,
                            help="the path of ckpt file")

    # Model related arguments
    parser.add_argument('--weight_loss', default=True,
                            help="if use weight loss")
    parser.add_argument('--use_trained', default=False,
                            help="if use trained model")
    parser.add_argument('--backbone', default='Deeplab50_CLASS_INW',
                            help="backbone of encoder")
    parser.add_argument('--bn_momentum', type=float, default=0.1,
                            help="batch normalization momentum")
    parser.add_argument('--imagenet_pretrained', type=str2bool, default=True,
                            help="whether apply imagenet pretrained weights")
    parser.add_argument('--pretrained_ckpt_file', type=str, default=None,
                            help="whether apply pretrained checkpoint")
    parser.add_argument('--continue_training', type=str2bool, default=False,
                            help="whether to continue training ")
    parser.add_argument('--show_num_images', type=int, default=2,
                            help="show how many images during validate")

    # train related arguments
    parser.add_argument('--seed', default=12345, type=int,
                            help='random seed')
    parser.add_argument('--gpu', type=str, default="0",
                            help=" the num of gpu")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                            help='input batch size')
    parser.add_argument('--alpha', default=0.3, type=int,
                            help='input mix alpha')


    parser.add_argument('--val_dataset', type=str, default='SimResist',
                            help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
    parser.add_argument('--base_size', default="256,256", type=str,
                            help='crop size of image')
    parser.add_argument('--crop_size', default="256,256", type=str,
                            help='base size of image')
    parser.add_argument('--target_base_size', default="256,256", type=str,
                            help='crop size of target image')
    parser.add_argument('--target_crop_size', default="256,256", type=str,
                            help='base size of target image')

    parser.add_argument('--data_loader_workers', default=1, type=int,
                            help='num_workers of Dataloader')
    parser.add_argument('--pin_memory', default=2, type=int,
                            help='pin_memory of Dataloader')
    parser.add_argument('--split', type=str, default='train',
                            help="choose from train/val/test/trainval/all")
    parser.add_argument('--random_mirror', default=True, type=str2bool,
                            help='add random_mirror')
    parser.add_argument('--random_crop', default=False, type=str2bool,
                            help='add random_crop')
    parser.add_argument('--resize', default=True, type=str2bool,
                            help='resize')
    parser.add_argument('--gaussian_blur', default=True, type=str2bool,
                            help='add gaussian_blur')
    parser.add_argument('--numpy_transform', default=True, type=str2bool,
                            help='image transform with numpy style')
    parser.add_argument('--color_jitter', default=True, type=str2bool,
                            help='image transform with numpy style')

    # optimization related arguments

    parser.add_argument('--freeze_bn', type=str2bool, default=False,
                            help="whether freeze BatchNormalization")

    parser.add_argument('--iter_max', type=int, default=200000,
                            help="the maxinum of iteration")
    parser.add_argument('--iter_stop', type=int, default=200000,
                            help="the early stop step")
    parser.add_argument('--each_epoch_iters', default=1000,
                            help="the path of ckpt file")
    parser.add_argument('--poly_power', type=float, default=0.9,
                            help="poly_power")
    parser.add_argument('--selected_classes', default=[0, 1],
                            help="poly_power")


    # multi-level output

    parser.add_argument('--multi', default=False, type=str2bool,
                            help='output model middle feature')
    parser.add_argument('--lambda_seg', type=float, default=0.1,
                            help="lambda_seg of middle output")

    return parser
