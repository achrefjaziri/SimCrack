"""Main script to train semantic segmentation models
Run the following command: python train.py --arch_name unet --dataset sim_Resist
"""
import numpy as np
from datetime import datetime
import os, logging
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from lib.models.unet import UNet
from lib.models.munet import SegPMIUNet,MultiUNet
from lib.dataloaders.sim_dataloader import SimDataloader
from lib.dataloaders.multi_crack_set_dataloader import MultiSetDataloader
from lib.training.train import train_model
from lib.training.validate import validate_model
from lib.utils.save_history import save_models
from lib.utils.custom_depth_loss import DepthLoss
from lib.arg_parser.general_args import parse_args


def train_and_val(args, model, train_loader, gpu, validation_loader, writer, current_dir):
    # img, label = next(iter(train_loader))
    # Loss function
    #class_weights = torch.FloatTensor([1,50]).to(gpu)
    loss_functions = {
        'SEG': nn.CrossEntropyLoss(),
        #'DEPTH': DepthLoss().to(gpu),
        'DEPTH': nn.MSELoss().to(gpu),
        'NRM': [nn.CosineEmbeddingLoss().to(gpu), nn.MSELoss().to(gpu)]
    }
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # gamma = decaying factor
    scheduler = StepLR(optimizer, step_size=2, gamma=0.96)

    if args.resume:
        start_epoch= args.res_epoch
    else:
        start_epoch = 0

    best_acc =0

    with torch.autograd.set_detect_anomaly(True):

        for epoch in range(start_epoch, args.num_epochs + start_epoch):
            train_loader.sampler.set_epoch(epoch)
            # Print Learning Rate
            # Print Learning Rate
            print('Epoch:', epoch + 1, 'LR:', scheduler.get_last_lr())
            logging.info(f'Epoch: {epoch + 1} , LR: {scheduler.get_last_lr()}')
            # train the model
            results = train_model(model, train_loader, loss_functions, optimizer,batch_size=args.batch_size,
                                  epoch=epoch,
                                  writer=writer, configs=args, gpu=gpu)
            train_acc, train_loss = results['Accuracy'], results['Loss']

            logging.info(f'Epoch {epoch + 1}, Train loss: {train_loss}, Train acc {train_acc}')
            print(epoch,train_loss,train_acc)

            results_val = validate_model(model, validation_loader, loss_functions, epoch + 1, writer,
                                         storage_directory=os.path.join(current_dir, "val_examples"), config=args,gpu=gpu)
            #loss_funcs, epoch, writer, storage_directory='prediction', config=None, gpu='cpu'
            print('Epoch', str(epoch + 1), 'Train loss:', train_loss, "Train acc", train_acc, 'Validation loss:',
                  results_val["Loss"], "Validation acc", results_val["Accuracy"])
            logging.info(
                f'Epoch {epoch + 1}, Validation loss {results_val["Loss"]}, Validation acc {results_val["Accuracy"]}')

            is_best = results_val['Accuracy'] > best_acc
            best_acc = max(results_val['Accuracy'], best_acc)

            # Decay Learning Rate
            scheduler.step()

            if epoch > 0:
                if is_best:
                    logging.info(f'Saving new best model {results["Accuracy"]} in epoch {epoch + 1}')
                    save_models(model, optimizer, current_dir, epoch + 1)


            print(f'Epoch {epoch + 1}, Train loss mean: {"%.4f" % np.mean(train_loss)},Loss per layer: {train_loss}')
            # print('Logging for epoch', epoch,train_loss)
            logging.info(
                f'Epoch {epoch + 1}, Train loss mean: {"%.4f" % np.mean(train_loss)},Loss per layer: {train_loss}')



    dist.destroy_process_group()
    writer.close()


def main(gpu, args, current_dir):
    # Set up the process groups
    print('Setup...')
    ############################################################
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    torch.cuda.set_device(gpu)

    ############################################################
    # load model
    print('Loading Model...')
    # TODO add the other models
    if args.arch_name=='unet' or args.arch_name=='pmiunet' :
        model = UNet(args.input_ch, args.num_classes)
    elif args.arch_name=='munet_pmi':
        model = SegPMIUNet(args.input_ch,args.num_classes)



    if args.resume:
        print("resuming training...")
        logging.info('LOADING MODEL: ' + args.model_path)
        if os.path.isfile(os.path.join(args.model_path, 'best_model.pth')):
            print("=> loading checkpoint '{}'".format(os.path.join(args.model_path, 'best_model.pth')))
            checkpoint = torch.load(os.path.join(args.model_path, 'best_model.pth'))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) successfully"
                  .format(os.path.join(args.model_path, 'best_model.pth'), checkpoint['epoch']))

    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    print('Creating Logger...')

    logging.basicConfig(
        handlers=[logging.FileHandler(filename=os.path.join(current_dir, f'run_history_{gpu}.log'),  # args.model_path
                                      encoding='utf-8', mode='a+')],
        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
        datefmt="%F %A %T",
        level=logging.INFO)
    logging.info(f''' Training parameters:
                    Architecture:  {args.arch_name}
                    Epochs:          {args.num_epochs}
                    Batch size:      {args.batch_size}
                    Learning rate:   {args.lr}
                    Weight decay:   {args.weight_decay}
                    Dataset:   {args.dataset}
                    Resize Crop Input: {args.resize_crop_input}
                    Input size: {args.input_size}
                    Resize size: {args.resize_size}
                    Input Channels: {args.input_ch}
                    PMI Neighbour Size: {args.neighbour_size}
                    PMI Phi Value: {args.phi_value}
                    PMI Hist Eq: {args.histequalize_pmi}
                    Custom Message: {args.m}
                ''')
    if args.resume:
        print("Resuming...")
        logging.info(f'Model Loaded from {args.model_path} Epoch: {args.res_epoch}')

    print('creating Tensorboard')
    model_id = os.path.basename(current_dir)
    tensorboard_dir = os.path.join(args.save_dir, 'tensorboard',
                                   'runs', args.dataset,
                                   args.arch_name + '-' + model_id)  # datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    writer = SummaryWriter(log_dir=tensorboard_dir,
                           comment=f'Model Id {model_id},Architecture {args.arch_name}, Training Set {args.dataset}')

    print('Loading Data...')

    # Data Loader
    if args.dataset == 'SimResist':
        training_set = SimDataloader(args, mode="train")
        validation_set = SimDataloader(args, mode="val")
    else:
        training_set = MultiSetDataloader(args, mode="train")
        validation_set = MultiSetDataloader(args, mode="test")

    validation_loader = \
        torch.utils.data.DataLoader(dataset=validation_set,
                                    num_workers=16, batch_size=args.batch_size, shuffle=False)

    sampler = DistributedSampler(training_set, num_replicas=args.world_size, rank=rank, shuffle=True,
                                 drop_last=False)
    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        sampler=sampler)

    print('Start training...')
    try:
        # Train the model
        train_and_val(args, model, train_loader, gpu, validation_loader, writer, current_dir)

    except KeyboardInterrupt:
        print("Training got interrupted.")


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parse_args()

    #########################################################
    if args.device != 'cpu':
        args.world_size = args.gpus * args.nodes
        print(args.world_size)

    os.environ['MASTER_ADDR'] = '127.0.0.1'  #
    os.environ['MASTER_PORT'] = args.port  # '6009'

    if args.resume:
        model_id = os.path.basename(args.model_path)
    else:
        model_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    current_dir = os.path.join(args.save_dir, args.arch_name, args.dataset, model_id)
    if not os.path.isdir(current_dir):
        os.makedirs(current_dir)

    mp.spawn(
        main,
        args=(args, current_dir,),
        nprocs=args.gpus
    )  #########################################################
