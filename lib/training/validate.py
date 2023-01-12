import torch
from sklearn.metrics import f1_score
import numpy as np
import os
import cv2


def validate_model(model, data_test, loss_funcs, epoch, writer, storage_directory='prediction', config=None, gpu='cpu'):
    """
    Test a model
    Args:
        :param model: the model to be tested
        :param data_test: (DataLoader) validation set
        :param loss_funcs: a dictionary of loss functions
        :param epoch: current epoch
        :param writer: Tensorboard writer
        :param storage_directory: where to store predictions
        :param configs: config parser
        :param gpu:  current gpz
    """
    model.eval()
    total_loss = 0
    total_f1 = 0
    print("Starting evaluation..")
    for batch, data in enumerate(data_test):
        with torch.no_grad():
            image = data['input'].to(gpu)
            mask = data['gt'].to(gpu)
            if config.arch_name == 'munet':
                # we only care about the segmentation output here
                _, output, _ = model(image)
            elif config.arch_name=='munet_pmi':
                output, _ = model(image)
            elif config.arch_name=='cons_unet':
                pmi_maps = data['pmi_map'].to(gpu)
                output, output_pmi,_ = model(image,pmi_maps)
            else:
                output = model(image)
            loss = loss_funcs['SEG'](output, mask)
            prediction = torch.argmax(output, dim=1).float().detach().cpu().numpy()

            if config.arch_name=='cons_unet' and config.fuse_predictions:
                loss_pmi = loss_funcs['SEG'](output_pmi, mask)
                prediction_pmi = torch.argmax(output_pmi, dim=1).float().detach().cpu().numpy()
                loss += loss_pmi
                prediction = np.add(prediction, prediction_pmi)
                prediction = prediction > 0.5

            total_loss += loss.item()

            acc_val = f1_score(mask.detach().cpu().numpy().ravel(), prediction.ravel())

            total_f1 += acc_val

            if config.save_val_examples and (batch % config.vis_freq == 0):
                print('printing images')
                im_name = os.path.basename(data['path'][0])
                desired_path = os.path.join(storage_directory, f'epoch_{epoch}', "crack_prediction")
                if not os.path.exists(desired_path):
                    os.makedirs(desired_path)

                cv2.imwrite(desired_path + str(im_name), np.squeeze(prediction[0].cpu().data.numpy()) * 255)

            torch.cuda.empty_cache()
    print("Evaluation done..")

    writer.add_scalar('Validation/Loss', total_loss / (batch + 1), epoch)
    writer.add_scalar('Validation/Accuracy', total_f1 / (batch + 1), epoch)

    results = {'epoch': epoch,
               'Loss': total_loss / (batch + 1),
               'Accuracy': total_f1 / (batch + 1),
               }

    return results
