'''
Evaluate different models on crack datasets. The segmentation results will be then saved in
'''
import logging
import os
import torch
from torch.utils.data import DataLoader
from lib.arg_parser.general_args import parse_args
from lib.models.unet import UNet
from lib.models.munet import SegPMIUNet, MultiUNet
from lib.models.consnet import AttU_Net, ConsNet
from lib.dataloaders.sim_dataloader import SimDataloader
from lib.dataloaders.real_dataloader import RealDataloader
from lib.dataloaders.multi_crack_set_dataloader import MultiSetDataloader
from lib.eval.evaluation_scripts import eval_model_patchwise, eval_model


def predict_cracks(args):
    # create folder
    save_dir = os.path.join(args.save_dir, "eval_outputs", args.dataset, args.arch_name,
                            os.path.basename(args.model_path))
    print("Results will be saved in the following directory:", save_dir)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    logging.basicConfig(handlers=[logging.FileHandler(filename=os.path.join(save_dir, 'run_history.log'),
                                                      encoding='utf-8', mode='a+')],
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%F %A %T",
                        level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info('Model Path:' + args.model_path)

    print('Using', args.arch_name)

    if args.arch_name == 'unet' or args.arch_name == 'pmiunet':
        print('loading unet')
        model = UNet(args.input_ch, args.num_classes)
    elif args.arch_name == 'munet':
        model = MultiUNet(args.input_ch, args.num_classes)
    elif args.arch_name == 'munet_pmi':
        model = SegPMIUNet(args.input_ch, args.num_classes)
    elif args.arch_name=='att_unet':
        model = AttU_Net(args.input_ch,args.num_classes)
    elif args.arch_name == 'cons_unet':
        print('args',args.cons_loss)
        model = ConsNet(args.input_ch, args.num_classes, att=args.att_connection, consistent_features=args.cons_loss,
                        img_size=args.input_size)


    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()
    logging.info('LOADING: ' + args.model_path + 'best_model.pth.tar')
    full_path = os.path.join(args.model_path, 'best_model.pth.tar')
    if os.path.isfile(full_path):
        print("=> loading checkpoint '{}'".format(full_path))
        checkpoint = torch.load(full_path, map_location=device)
        training_epochs = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(full_path, checkpoint['epoch']))

    print('Loading data..')
    # Data Loader
    if args.dataset == 'SimResist':
        test_set = SimDataloader(args, mode="val")
    elif args.dataset == 'RealResist':
        test_set = RealDataloader(args)
    elif args.dataset == 'MultiSet':
        test_set = MultiSetDataloader(args, mode="test")
    else:
        print('Choose valid dataset!')
        exit()

    test_loader = \
        torch.utils.data.DataLoader(dataset=test_set,
                                    num_workers=16, batch_size=1, shuffle=False)

    logging.info(f''' Evaluation parameters:
              Architecture:  {args.arch_name}
                    Training Epochs: {training_epochs}
                    Dataset:   {args.dataset}
                    Patchwise eval: {args.patchwise_eval}
                    Resize Crop Input: {args.resize_crop_input}
                    Input Channels:  {args.input_ch}
                    Input size: {args.input_size}
                    Resize size: {args.resize_size}
                    Custom Message: {args.m}
        ''')
    if args.patchwise_eval:
        eval_model_patchwise(model, test_loader,
                             storage_directory=os.path.join(save_dir, "segmentations"), args=args, device=device)
    else:
        eval_model(model, test_loader,
                   storage_directory=os.path.join(save_dir, "segmentations"), args=args, device=device)


if __name__ == "__main__":
    print(" Evaluating semantic segmentation model...")
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    args = parse_args()
    predict_cracks(args)
