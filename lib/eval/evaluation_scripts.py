"""
Create semantic predictions using a model and a test set. The predictions are then saved in the storage_directory.
"""

import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from lib.utils.tensor_utils import remove_padding,get_padding_values


def eval_model_patchwise(model, data_test, storage_directory='prediction',args= None,device='cpu'):
    """
    Test a model patchwise: create patches of size args.patch_size and then downsize the patches to size args.input_size
    Args:
        :param model: the model to be tested
        :param data_test: (DataLoader): validation set
        :param storage_directory: where to store predictions
        :param args: Model configuration
        :param device: memory device (cpu or cuda)
    """
    model.eval()
    patch_size = args.patch_size
    input_size = args.input_size

    print("Starting evaluation..")
    for batch, data in enumerate(data_test):
        # print(image_v.shape, mask_v.shape)
        with torch.no_grad():
            img = data['input'][0].to(device)

            #get padding values for each dimension
            x1, y1 = get_padding_values(img.size(1), patch_size)
            x2, y2 = get_padding_values(img.size(2), patch_size)
            #pad images
            img = F.pad(img, (x2, y2, x1, y1))

            #change the first unfold to 3 ,3 if in grey scale image else rgb img
            if args.input_ch==1:
                patches = img.data.unfold(0, 1, 1).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            else:
                patches = img.data.unfold(0, 3, 3).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

            #save shape of images patches
            shape_of_img_patches = patches.data.cpu().numpy().shape
            #print('shape of image patches', shape_of_img_patches)

            #flatten patches
            patches = torch.flatten(patches, start_dim=0, end_dim=2)
            #print('shape eval patches', patches.shape)
            results = []
            #Start evaluating patches of an images
            for i in range(patches.shape[0]):
                # print(i)
                current_img = torch.unsqueeze(patches[i], dim=0)
                #Downsize image if needd
                current_img = F.interpolate(current_img, size=(input_size, input_size), mode='bicubic', align_corners=False)

                if args.arch_name =='munet':
                    y, seg_output, x = model(current_img)

                else:
                    seg_output = model(current_img)

                #append output of a current patch
                out_np = torch.squeeze(seg_output).detach().cpu().numpy()
                out_np_channel_1 = cv2.resize(out_np[0],(patch_size, patch_size))
                out_np_channel_2 = cv2.resize(out_np[1],(patch_size, patch_size))
                out_np = np.array([out_np_channel_1,out_np_channel_2])
                results.append(out_np)
            out_image = np.asarray(results)

            #Reshape patches before stiching them up
            out_image = np.reshape(out_image, (
            shape_of_img_patches[0], shape_of_img_patches[1], shape_of_img_patches[2], 2, shape_of_img_patches[4], shape_of_img_patches[5]))

            #Stich image backup again
            stitched_out = torch.from_numpy(out_image).permute(0, 3, 1, 4, 2, 5).contiguous().view(
                [2, img.shape[1], img.shape[2]])
            stitched_out = remove_padding(stitched_out.detach().cpu().numpy(), x1, y1, x2, y2)

            #print('Stiched output',stitched_out.shape)
            #Arg max to get the class in the segmentation map from softmax outputs
            stitched_out = np.argmax(stitched_out, axis=0)
            img_name = os.path.basename(data['path'][0])
            if not os.path.exists(storage_directory):
                    os.makedirs(storage_directory)
            cv2.imwrite(os.path.join(storage_directory, str(img_name)) , stitched_out * 255)

            torch.cuda.empty_cache()

def eval_model(model, data_test, storage_directory='prediction', args = None,device='cpu'):
    """
    Test a model
    Args:
        :param model: the model to be tested
        :param data_test: (DataLoader): validation set
        :param storage_directory: where to store predictions
        :param args: Model configuration
        :param device: memory device (cpu or cuda)
    """
    model.eval()

    print("Starting evaluation..")
    for batch, data in enumerate(data_test):
        # print(image_v.shape, mask_v.shape)
        with torch.no_grad():
            image = data['input'].to(device)
            if args.arch_name == 'munet':
                y,output,x = model(image)

            else:
                output = model(image)
            prediction = torch.argmax(output, dim=1).float()

            im_name = os.path.basename(data['path'][0])
            if not os.path.exists(storage_directory):
                    os.makedirs(storage_directory)

            cv2.imwrite(os.path.join(storage_directory, str(im_name)), np.squeeze(prediction[0].cpu().data.numpy()) * 255)

            torch.cuda.empty_cache()
    print("done..")


