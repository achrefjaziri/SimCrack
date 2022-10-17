def parse_pmi_args(parser):
    # General Training parameters

    parser.add_argument('--phi_value', type=float, default=1.75,
                        help='')

    parser.add_argument('--neighbour_size', type=int, default=5,
                        help='')

    parser.add_argument('--pmi_dir', default="/data/resist_data/pmi_maps",
        help="Port specification needed for distributed multi-gpu training")

    return parser
