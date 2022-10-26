'''
Code taken from https://github.com/Yuki-11/CSSR and adapted to evaluate the models on our dataset.

'''
import argparse
import datetime
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler
from torch.multiprocessing import Pool, Process, set_start_method
from baseline.config import cfg
from baseline.modeling.build_model import Model, InvModel
from baseline.data.transforms.data_preprocess import TestTransforms
from baseline.data.crack_dataset import CrackDataSetTest
from baseline.engine.inference import inference_for_ss
from baseline.utils.misc import fix_model_state_dict, send_line_notify
from baseline.data.transforms.transforms import FactorResize
from torch.multiprocessing import Pool, Process, set_start_method
import cv2
import torch.nn.functional as F



def get_padding_values(h,patch_size):
    n = h // patch_size
    if (h % patch_size) ==0:
        return 0,0
    else:
        additional_pixels = (n+1)* patch_size - h
        if (additional_pixels % 2) ==0:
            return additional_pixels//2,additional_pixels//2
        else:
            return (additional_pixels // 2)+1, (additional_pixels // 2)



def load_img(path_pred, rgb_mode=True):
    img = cv2.imread(path_pred)  # cv2.IMREAD_GRAYSCALE
    if rgb_mode:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #img = cv2.resize(img, (448 //4, 448//4))
    # plt.imshow(np.squeeze(img[:,:,0])) #,cmap ='gray'
    # plt.show() #, cv2.IMREAD_GRAYSCALE
    # plt.close()

    return img

def test(args, cfg):
    device = torch.device(cfg.DEVICE)
    # model = Model(cfg).to(device)
    if cfg.MODEL.SR_SEG_INV:
        print('inv model')
        model = InvModel(cfg).to(device)
        print(
            f'------------Model Architecture-------------\n\n<Network SS>\n{model.segmentation_model}\n\n<Network SR>\n{model.sr_model}')
    else:
        print('non inv model')
        model = Model(cfg).to(device)
        print(
            f'------------Model Architecture-------------\n\n<Network SR>\n{model.sr_model}\n\n<Network SS>\n{model.segmentation_model}')

    model.load_state_dict(
        fix_model_state_dict(torch.load(args.trained_model, map_location=lambda storage, loc: storage)))
    model.eval()

    path_pred = '/data/resist_data/datasets/resist_set/images/strymonas_0000119_0_0.png'  # strymonas_0000511_0_1688.png

    img_rgb = load_img(path_pred, True)
    print(img_rgb.shape)

    img = torch.Tensor(img_rgb.transpose(2,0,1)).to('cuda')

    patch_size = 448
    input_size = 448 // 4
    x1, y1 = get_padding_values(img.size(1), patch_size)
    x2, y2 = get_padding_values(img.size(2), patch_size)
    # pad images
    img = F.pad(img, (x2, y2, x1, y1))

    print(img.shape)

    patches = img.data.unfold(0, 3, 3).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

    # save shape of images patches
    shape_of_img_patches = patches.data.cpu().numpy().shape
    # print('shape of image patches', shape_of_img_patches)

    # flatten patches
    patches = torch.flatten(patches, start_dim=0, end_dim=2)
    # print('shape eval patches', patches.shape)
    results = []
    # Start evaluating patches of an images
    for i in range(patches.shape[0]):
        # print(i)
        current_img = torch.unsqueeze(patches[i], dim=0)
        # Downsize image if needd
        current_img = F.interpolate(current_img, size=(input_size, input_size), mode='bicubic', align_corners=False)

        m = model(current_img)
        print(m[0].shape, m[1].shape)
        seg = m[1].detach().cpu().numpy()
        plt.imshow(seg[0, 0])
        plt.show()



    '''
    print('Loading Datasets...')
    test_transforms = TestTransforms(cfg)
    sr_transforms = FactorResize(cfg.MODEL.SCALE_FACTOR)
    test_dataset = CrackDataSetTest(cfg, cfg.DATASET.TEST_IMAGE_DIR, cfg.DATASET.TEST_MASK_DIR,
                                    transforms=test_transforms, sr_transforms=sr_transforms)
    sampler = SequentialSampler(test_dataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)

    if args.num_gpus > 1:
        # for k in models.keys():
        #     device_ids = list(range(args.num_gpus))
        #     print("device_ids:",device_ids)
        #     # models[k] = torch.nn.DataParallel(models[k], device_ids=device_ids)
        device_ids = list(range(args.num_gpus))
        print("device_ids:", device_ids)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    with torch.no_grad():
        inference_for_ss(args, cfg, model, test_loader)

    '''



def main():
    parser = argparse.ArgumentParser(description='Crack Segmentation with Super Resolution(CSSR)')
    parser.add_argument('test_dir', type=str, default=None)
    parser.add_argument('iteration', type=int, default=None)

    parser.add_argument('--output_dirname', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None, metavar='FILE')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_gpus', type=int, default=6)
    parser.add_argument('--test_aiu', type=bool, default=True)
    parser.add_argument('--trained_model', type=str, default=None)
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--wandb_prj_name', type=str, default="CSSR_test")

    args = parser.parse_args()

    check_args = [('config_file', f'{args.test_dir}config.yaml'),
                  ('output_dirname', f'{args.test_dir}eval_AIU/iter_{args.iteration}'),
                  ('trained_model', f'{args.test_dir}model/iteration_{args.iteration}.pth'),
                  ]

    for check_arg in check_args:
        arg_name = f'args.{check_arg[0]}'
        if exec(arg_name) == None:
            exec(f'{arg_name} = "{check_arg[1]}"')

    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True

    if len(args.config_file) > 0:
        print('Configration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)

    cfg.OUTPUT_DIR = args.output_dirname

    cfg.freeze()

    print('Running with config:\n{}'.format(cfg))

    test(args, cfg)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] ='2'
    set_start_method('spawn')
    main()
