import argparse

def parse_common_evaluation_args(parser):
    parser.add_argument(
        "--save_dir",
        default="./workspace",
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
    parser.add_argument('--gt_path', default='/data/resist_data/datasets/sim_crack/test/gts',
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


    return parser
