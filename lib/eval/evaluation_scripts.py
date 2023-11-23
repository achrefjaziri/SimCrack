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

            if args.arch_name=='cons_unet':
                img_pmi = data['pmi_map'][0].to(device)
                img_pmi = F.pad(img_pmi, (x2, y2, x1, y1))
                patches_pmi = img_pmi.data.unfold(0, 1, 1).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
                patches_pmi = torch.flatten(patches_pmi, start_dim=0, end_dim=2)
                results_pmi = []

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
                #print(i)
                current_img = torch.unsqueeze(patches[i], dim=0)
                #Downsize image if needd
                current_img = F.interpolate(current_img, size=(input_size, input_size), mode='bicubic', align_corners=False)
                if args.arch_name =='munet':
                    _, seg_output, _ = model(current_img)
                elif args.arch_name =='san_saw':
                    pred = model(current_img)
                    seg_output = pred[2]
                elif args.arch_name =='munet_pmi':
                    seg_output, _ = model(current_img)
                elif args.arch_name == 'cons_unet' or args.arch_name == '2unet':
                    if args.arch_name == 'cons_unet':
                        current_img_pmi = torch.unsqueeze(patches_pmi[i], dim=0)
                        # Downsize image if needd
                        current_img_pmi = F.interpolate(current_img_pmi, size=(input_size, input_size), mode='bicubic',
                                                align_corners=False)
                        seg_output,seg_output_pmi, _ = model(current_img,current_img_pmi)
                    else:
                        seg_output,seg_output_pmi, _ = model(current_img,current_img)

                    out_pmi_np = torch.squeeze(seg_output_pmi).detach().cpu().numpy()
                    out_pmi_np_channel_1 = cv2.resize(out_pmi_np[0], (patch_size, patch_size))
                    out_pmi_np_channel_2 = cv2.resize(out_pmi_np[1], (patch_size, patch_size))
                    out_pmi_np = np.array([out_pmi_np_channel_1, out_pmi_np_channel_2])
                    results_pmi.append(out_pmi_np)
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

            if (args.arch_name == 'cons_unet' or args.arch_name == '2unet') and args.fuse_predictions: #we need to restitch the outputs for the case of the second output map
                out_image = np.asarray(results_pmi)
                # Reshape patches before stiching them up
                out_image = np.reshape(out_image, (
                    shape_of_img_patches[0], shape_of_img_patches[1], shape_of_img_patches[2], 2,
                    shape_of_img_patches[4], shape_of_img_patches[5]))

                # Stich image backup again
                stitched_out_pmi = torch.from_numpy(out_image).permute(0, 3, 1, 4, 2, 5).contiguous().view(
                    [2, img.shape[1], img.shape[2]])
                stitched_out_pmi = remove_padding(stitched_out_pmi.detach().cpu().numpy(), x1, y1, x2, y2)

                # Arg max to get the class in the segmentation map from softmax outputs
                stitched_out_pmi = np.argmax(stitched_out_pmi, axis=0)
                stitched_out = np.add(stitched_out, stitched_out_pmi)
                stitched_out = stitched_out > 0.5
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
                _,output,_ = model(image)
            elif args.arch_name == 'munet_pmi':
                output, _ = model(image)
            elif args.arch_name == 'cons_unet':
                pmi_maps = data['pmi_map'].to(device)
                output,output_pmi, _ = model(image,pmi_maps)
            elif args.arch_name == '2unet':
                output,output_pmi, _ = model(image,image)
            else:
                output = model(image)


            prediction = torch.argmax(output, dim=1).float().float().detach().cpu().numpy()
            if (args.arch_name=='cons_unet' or args.arch_name == '2unet') and args.fuse_predictions:
                prediction_pmi = torch.argmax(output_pmi, dim=1).float().detach().cpu().numpy()
                prediction = np.add(prediction, prediction_pmi)
                prediction = prediction > 0.5

            im_name = os.path.basename(data['path'][0])
            if not os.path.exists(storage_directory):
                    os.makedirs(storage_directory)

            cv2.imwrite(os.path.join(storage_directory, str(im_name)), np.squeeze(prediction[0]) * 255)

            torch.cuda.empty_cache()
    print("done..")


