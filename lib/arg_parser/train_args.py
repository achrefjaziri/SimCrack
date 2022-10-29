import argparse

def parse_train_args(parser):
    # General Training parameters
    parser.add_argument('--no_batch_norm', default=False,
                        help='No batch norm if true')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout after each nonlinearity (default: 0.0)')
    parser.add_argument('--nonlin', default='relu',
                        help='nonlinearity, relu or leakyrelu (default: relu)')
    parser.add_argument('--optim', default='adam',
                        help='optimizer to be used')
    parser.add_argument('--beta', default=0.1, type=float,
                        help='beta value for ADAM optimizer')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')

    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="weight decay or l2-penalty on weights (for ADAM optimiser, default = 0., i.e. no l2-penalty)"
    )

    parser.add_argument(
        "--weight_init",
        default=False,
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )


    parser.add_argument(
        "--histequalize_pmi",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use histogram equalization on PMI values"
    )

    parser.add_argument(
        "--augment",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use data Augmentation during training"
    )


    return parser
