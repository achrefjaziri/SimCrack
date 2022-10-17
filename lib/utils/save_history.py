import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_eval_history(header, new_row, file_name):
    """ export data to csv format
    Args:
        :param header: (Dict) headers of the column
        :param new_row: (Dict) contains value for the current row
        :param file_name: path of the csv file
    """
    # make new folder
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    df = pd.DataFrame(data=header)
    df = pd.concat([df, pd.DataFrame.from_records([new_row])])


    with open(file_name, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell() == 0)

def save_models(model, optimizer, path, epoch):
    """Save model to given path
    Args:
        :param model: model to be saved
        :param optimizer: parameters of the optimizer are also saved
        :param path: path that the model
        :param epoch: the current epoch of the model
    """
    checkpoint = {'epoch': epoch +1,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(checkpoint, path+"/best_model.pth.tar")


def save_prediction_image_seg(img, mask, im_name, save_folder_name, acc_val, h_euc, h_rbf, wcn, f1, theta):
    """save images to save_path
        Args:
            stacked_img (numpy): stacked cropped images
            save_folder_name (str): saving folder name
    """
    fig = plt.figure(figsize=(5, 5))
    textstr = ' IOU=%.2f\n h_euc=%.2f\n h_rbf=%.2f\n WCN=%.2f\n f1=%.2f\n' % (acc_val, h_euc, h_rbf, wcn, f1)
    # ax enables access to manipulate each of subplots
    ax = []
    # orig = np.transpose(origimg.detach().cpu().numpy(), (1, 2, 0))
    # orig = cv2.resize(orig, dsize=(2048, 2048), interpolation=cv2.INTER_CUBIC)
    # ax.append(fig.add_subplot(1, 3, 1))
    # ax[-1].set_title("Original")  # set title
    # plt.axis('off')
    # plt.imshow(np.squeeze(orig))

    # groundTruth = np.transpose(mask.detach().cpu().numpy(), (1, 2, 0))
    # groundTruth = cv2.resize(mask.cpu().data.numpy(), dsize=(2048, 2048), interpolation=cv2.INTER_CUBIC)
    ax.append(fig.add_subplot(1, 2, 1))
    ax[-1].set_title("Ground Truth")  # set title
    plt.axis('off')
    plt.imshow(mask)

    # dilated_gt = binary_dilation(mask, iterations=theta)
    # print(dilated_gt.shape)
    # ax.append(fig.add_subplot(1, 3, 2))
    # ax[-1].set_title("Dilated Gt")  # set title
    # plt.axis('off')
    # plt.imshow(np.squeeze(dilated_gt))
    # out = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    # out = cv2.resize(img, dsize=(2048, 2048), interpolation=cv2.INTER_CUBIC)
    ax.append(fig.add_subplot(1, 2, 2))
    ax[-1].set_title("Prediction")  # set title
    plt.axis('off')
    plt.text(600, 600, textstr, fontsize=9)
    plt.imshow(np.squeeze(img))

    # Create the path if it does not exist
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)
    # Save Image!
    export_name = str(im_name) + '.png'
    plt.savefig(save_folder_name + str(im_name), bbox_inches='tight')
    plt.close()