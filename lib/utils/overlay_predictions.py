'''
Script to overlay images and their predicted masks
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import random
def overlay(input_path, gt_path,unet_path, unetMulti_path,ours_path ,out_path,save_directory):
    '''function to overlay images'''
    im = cv2.imread(input_path)
    print("im loaded")
    #im = cv2.resize(im, (512, 512))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)




    gt_img = cv2.imread(gt_path, 1)
    #gt_img = cv2.resize(gt_img, (512, 512))
    gt_img = np.ma.masked_where(gt_img == 0, gt_img)
    print(gt_img.shape)

    mask_cropped = np.zeros((gt_img.shape[0],gt_img.shape[1]))
    count = 0
    while (np.count_nonzero(mask_cropped[:, :]) < 200) and (
            count < 50):
        y = random.randrange(988)
        x = random.randrange(332)
        mask_cropped = gt_img[x:x+512,y:y+512]
        count+=1

    gt_img = mask_cropped
    unet_img = cv2.imread(unet_path, 1)
    #unet_img = cv2.resize(unet_img, (512, 512))
    unet_img = np.ma.masked_where(unet_img == 0, unet_img)
    unet_img = unet_img[x:x+512,y:y+512]

    unetMulti_img = cv2.imread(unetMulti_path, 1)
    #unetMulti_img = cv2.resize(unetMulti_img, (512, 512))
    unetMulti_img = np.ma.masked_where(unetMulti_img == 0, unetMulti_img)

    ours_img = cv2.imread(ours_path, 1)
    #ours_img = cv2.resize(ours_img, (512, 512))
    ours_img = np.ma.masked_where(ours_img == 0, ours_img)
    ours_img = ours_img[x:x+512,y:y+512]
    im = im[x:x+512,y:y+512]
    plt.figure(figsize=(30, 30))
    plt.subplot(1, 4, 1)
    plt.axis('off')
    plt.imshow(im, 'gray', interpolation='none')
    #plt.gca().set_title('Input')

    plt.subplot(1, 4, 2)
    plt.imshow(im, 'gray', interpolation='none')
    plt.imshow(gt_img, 'jet', interpolation='none', alpha=0.6)
    plt.axis('off')
    #plt.gca().set_title('Ground Truth')


    plt.subplot(1, 4, 3)
    plt.imshow(im, 'gray', interpolation='none')
    plt.imshow(unet_img, 'jet', interpolation='none', alpha=0.6)
    plt.axis('off')
    #plt.gca().set_title('U-Net (Sim)')

    #plt.subplot(1, 5, 4)
    #plt.imshow(im, 'gray', interpolation='none')
    #plt.imshow(unetMulti_img, 'jet', interpolation='none', alpha=0.6)
    #plt.axis('off')
    #plt.gca().set_title('U-Net (Sim)')


    plt.subplot(1, 4, 4)
    plt.imshow(im, 'gray', interpolation='none')
    plt.imshow(ours_img, 'jet', interpolation='none', alpha=0.6)
    plt.axis('off')
    #plt.gca().set_title('Ours')


    plt.tight_layout()
    save_name = save_directory+'/'+ os.path.basename(out_path)
    plt.savefig(save_name,bbox_inches ='tight',
    pad_inches = 0)

if __name__ == '__main__':


    inputs_directory = '/data/resist_data/datasets/resist_set/images'


    predictions_directoryOurs = '/data/resist_data/SimCrack/workspace/eval_outputs/RealResist/cons_unet/2023-01-17_18-20-35/segmentations'
    predictions_directoryMultiSet = '/data/resist_data/SimCrack/workspace/eval_outputs/RealResist/unet/2022-10-19_15-23-09/segmentations'
    predictions_directoryUnet = '/data/resist_data/SimCrack/workspace/eval_outputs/RealResist/unet/2023-01-04_12-37-41/segmentations'
    predictions_directoryGT = '/data/resist_data/datasets/resist_set/gts'


    save_dir = '/data/resist_data/results/overlayed'


    inputs_list = glob.glob(str(inputs_directory) + "/*")
    gt_list = glob.glob(str(predictions_directoryGT) + "/*")
    ours_list = glob.glob(str(predictions_directoryOurs) + "/*")
    unet_list = glob.glob(str(predictions_directoryUnet) + "/*")
    unetMulti_list = glob.glob(str(predictions_directoryMultiSet) + "/*")


    gt_list.sort()
    ours_list.sort()
    unet_list.sort()
    unetMulti_list.sort()
    inputs_list.sort()

    print("lengths",len(inputs_list),len(unetMulti_list),len(unet_list),len(ours_list),len(gt_list))
    for i,input_path in enumerate(inputs_list):
        out_path = save_dir + '/'+os.path.basename(input_path)#.replace('.png','') +'.png' #change this to png sometimesS
        print('paths',input_path,gt_list[i],ours_list[i],input_path,save_dir)

        #try:
        overlay(input_path,gt_list[i],unet_list[i],unetMulti_list[i],ours_list[i], out_path, save_dir)
        print(f'image {i} save')
        #except:
        #    print(f'image {i} passed')

