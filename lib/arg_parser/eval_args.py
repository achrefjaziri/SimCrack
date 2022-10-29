import argparse

def parse_eval_args(parser):
    # Metrics that can be included during evaluation
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
                        help='folder containing ground truth maps') # /data/resist_data/datasets/resist_set/gts ,/data/resist_data/datasets/crack_segmentation_dataset/test/masks
    parser.add_argument('--pred_path', default='/home/ajaziri/resist_projects/SimCrack/workspace/eval_outputs/SimResist/unet/2022-09-22_09-39-19',
                        help='folder containing prediction maps. The basename of this folder will be used to name csv files')

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
        "--num_cpus",
        default=32,
        type=int,
        help="Number of cpu cores used for parallel evaluation",
    )

    return parser
