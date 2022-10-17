from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
import numpy as np


def train_model(model, data_train, loss_funcs,
                optimizer, batch_size, epoch, writer, configs, gpu):
    """Train the unet or munet models for one epoch and report training loss and accuracy values
    Args:
        :param model: the model to be trained
        :param data_train: (DataLoader) training set
        :param loss_funcs: a dictionary of loss function
        :param optimizer: The optimizer to be used
        :param batch_size: size of the training batch
        :param epoch: current epoch
        :param writer: Tensorboard writer
        :param configs: config parser
        :param gpu:  current gpz
    """
    model.train()
    optimizer.zero_grad()
    counter = 0
    total_loss = 0
    total_acc = 0
    total_loss_depth = 0
    total_loss_seg = 0
    total_loss_normal = 0
    print('LEN TRAIN',len(data_train))
    with tqdm(total=len(data_train), desc=f'Epoch {epoch + 1}/{configs.num_epochs}', unit='batch') as pbar:
        for batch, (data) in enumerate(data_train):
            counter += data['input'].shape[0]
            images = data['input'].to(gpu)
            masks = data['gt'].to(gpu)

            if configs.arch_name == 'munet':
                depths = data['depth'].to(gpu)
                nrms = data['normal'].to(gpu)
                y = torch.ones((masks.shape[0], masks.shape[0], masks.shape[1], masks.shape[1])).cuda() #used to evaluate cosine similarity
                output_nrm, output_seg, output_depth = model(images)
                loss_nrm = loss_funcs['NRM'][0](output_nrm, nrms, y) + loss_funcs['NRM'][1](output_nrm, nrms)
                loss_seg = loss_funcs['SEG'](output_seg, masks)
                loss_depth = loss_funcs['DEPTH'](output_depth, depths)
                print(batch, "Seg: ", loss_seg.item(), "depth: ", loss_depth.item(), "nrm: ", loss_nrm.item())
                if (loss_depth.item() == float("nan")) or (np.isnan(loss_depth.item())):
                    # Depth is skipped sometimes...  Some depth maps are corrupted.
                    loss = loss_seg + loss_nrm
                else:
                    loss = loss_seg + loss_depth + loss_nrm
                    total_loss_depth = total_loss_depth + loss_depth.item()

                loss.backward()
                total_loss = total_loss + loss.item()
                total_loss_seg = total_loss_seg + loss_seg.item()
                total_loss_normal = total_loss_normal + loss_nrm.item()
            else:

                output_seg = model(images)
                loss = loss_funcs['SEG'](output_seg, masks)
                loss.backward()
                total_loss = total_loss + loss.item()

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            prediction_map = torch.argmax(output_seg, dim=1).float()
            f1_acc = f1_score(masks.detach().cpu().numpy().ravel(), prediction_map.detach().cpu().numpy().ravel(),
                           average='binary')
            total_acc = total_acc + f1_acc
    writer.add_scalar('train/Loss', total_loss / (batch + 1), epoch)
    writer.add_scalar('train/Accuracy', total_acc / (batch + 1), epoch)
    if configs.arch_name == 'munet':
        writer.add_scalar('train/Loss_Seg', total_loss_seg / (batch + 1), epoch)
        writer.add_scalar('train/Loss_Nrm', total_loss_normal / (batch + 1), epoch)
        writer.add_scalar('train/Loss_Depth', total_loss_depth / (batch + 1), epoch)
    print('training epoch done..')
    results = {'epoch': epoch,
               'Loss': total_loss / (batch + 1),
               'Loss_Seg': total_loss_seg / (batch + 1),
               'Loss_Depth': total_loss_depth / (batch + 1),
               'Loss_Nrm': total_loss_normal / (batch + 1),
               'Accuracy': total_acc / (batch + 1)}
    return results
