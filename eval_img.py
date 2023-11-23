import numpy as np
from PIL import Image
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import argparse
from lib.models.unet import UNet
from lib.models.munet import SegPMIUNet, MultiUNet
from lib.models.consnet import AttU_Net, ConsNet
from lib.models.transunet import TransUNet
from lib.dataloaders.sim_dataloader import SimDataloader
from lib.dataloaders.real_dataloader import RealDataloader
from lib.dataloaders.multi_crack_set_dataloader import MultiSetDataloader
from lib.eval.evaluation_scripts import eval_model_patchwise, eval_model
from lib.utils.tensor_utils import remove_padding,get_padding_values
from create_pmi_weights import save_pmi_maps,save_patch_wise_pmi_maps
import cv2
from skimage import exposure
from torchvision import transforms
from PIL import Image, ImageOps



def evaluate_img_patchwise(img,input_size,patch_size,model,img_pmi=None):
    x1, y1 = get_padding_values(img.size(1), patch_size)
    x2, y2 = get_padding_values(img.size(2), patch_size)
    # pad images
    img = F.pad(img, (x2, y2, x1, y1))

    if args.arch_name == 'cons_unet':
        img_pmi = F.pad(img_pmi, (x2, y2, x1, y1))
        patches_pmi = img_pmi.data.unfold(0, 1, 1).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches_pmi = torch.flatten(patches_pmi, start_dim=0, end_dim=2)
        results_pmi = []

    if args.input_ch == 1:
        patches = img.data.unfold(0, 1, 1).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    else:
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
        current_img = torch.unsqueeze(patches[i], dim=0)
        # Downsize image if needd
        current_img = F.interpolate(current_img, size=(input_size, input_size), mode='bicubic', align_corners=False)
        if args.arch_name == 'munet':
            _, seg_output, _ = model(current_img)
        elif args.arch_name == 'munet_pmi':
            seg_output, _ = model(current_img)
        elif args.arch_name == 'cons_unet' or args.arch_name == '2unet':
            if args.arch_name == 'cons_unet':
                current_img_pmi = torch.unsqueeze(patches_pmi[i], dim=0)
                # Downsize image if needd
                current_img_pmi = F.interpolate(current_img_pmi, size=(input_size, input_size), mode='bicubic',
                                                align_corners=False)
                seg_output, seg_output_pmi, _ = model(current_img, current_img_pmi)
            else:
                seg_output, seg_output_pmi, _ = model(current_img, current_img)

            out_pmi_np = torch.squeeze(seg_output_pmi).detach().cpu().numpy()

            #out_for_patch = np.argmax(out_pmi_np,axis=0)
            #cv2.imwrite(f'/data/resist_data/SimCrack/tmp_patches/{i}.png', out_for_patch * 255)
            out_pmi_np_channel_1 = cv2.resize(out_pmi_np[0], (patch_size, patch_size))
            out_pmi_np_channel_2 = cv2.resize(out_pmi_np[1], (patch_size, patch_size))
            out_pmi_np = np.array([out_pmi_np_channel_1, out_pmi_np_channel_2])
            results_pmi.append(out_pmi_np)
        else:
            seg_output = model(current_img)
        # append output of a current patch
        out_np = torch.squeeze(seg_output).detach().cpu().numpy()
        out_np_channel_1 = cv2.resize(out_np[0], (patch_size, patch_size))
        out_np_channel_2 = cv2.resize(out_np[1], (patch_size, patch_size))
        out_np = np.array([out_np_channel_1, out_np_channel_2])
        results.append(out_np)
    out_image = np.asarray(results)

    # Reshape patches before stiching them up
    out_image = np.reshape(out_image, (
        shape_of_img_patches[0], shape_of_img_patches[1], shape_of_img_patches[2], 2, shape_of_img_patches[4],
        shape_of_img_patches[5]))

    # Stich image backup again
    stitched_out = torch.from_numpy(out_image).permute(0, 3, 1, 4, 2, 5).contiguous().view(
        [2, img.shape[1], img.shape[2]])
    stitched_out = remove_padding(stitched_out.detach().cpu().numpy(), x1, y1, x2, y2)

    # print('Stiched output',stitched_out.shape)
    # Arg max to get the class in the segmentation map from softmax outputs
    stitched_out = np.argmax(stitched_out, axis=0)

    if (
            args.arch_name == 'cons_unet' or args.arch_name == '2unet'):  # we need to restitch the outputs for the case of the second output map
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
    return stitched_out

# Load the image
def load_image(image_path):
    image = Image.open(image_path)
    return image #np.array(image)

# Preprocess the image for the model
def preprocess_image(image,skip_resize=False,input_size=256):
    if not skip_resize:
        transform = T.Compose([
            T.Resize((input_size,input_size)),  # Adjust the input size to match the model's requirements
        ])
        image = transform(image)
    #image = image[:3,:, :]
    #print('image after this',image.shape)
    #image_from_tensor = transforms.ToPILImage()(image)

    # Save the PIL image to a file
    #image_from_tensor.save('/data/resist_data/SimCrack/outs/output_image1.jpg')  # Replace 'output_image.jpg' with your desired output file path

    # delete alpha channel
    image = np.asarray(image)#.transpose(1,2,0)
    #print('dd',image.shape)
    image = image[:, :, :3]
    image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image).float()
    #print('img shape',image.shape)
    image_from_tensor = transforms.ToPILImage()(image)

    # Save the PIL image to a file
    image_from_tensor.save('/data/resist_data/SimCrack/outs/output_image.jpg')  # Replace 'output_image.jpg' with your desired output file path

    return image.unsqueeze(0)  # Add batch dimension

# Postprocess the model output to get the segmentation map
def postprocess_output(original_image,output):
    output = F.interpolate(output, size=(original_image.shape[0], original_image.shape[1]), mode='bilinear', align_corners=False)
    output = output.argmax(1)
    return output.squeeze().cpu().numpy()

def get_pmi_map(image,args,patchwise=False):
    if patchwise:
        input_img = image.transpose(2,0,1) #.transpose(0,1, 2, 0)
        img_name = 'tmp_img.npy'
        pmi_dir='/data/resist_data/SimCrack/tmp_pmi/' #TODO Create Directory if not available
        npy_path='/data/resist_data/SimCrack/tmp_pmi/tmp_img.npy'
        input_img = torch.Tensor(input_img)
        save_patch_wise_pmi_maps(input_img, args, pmi_dir, img_name)
        pmi_maps = np.load(npy_path)  # .transpose(1,2,0)
        if args.histequalize_pmi:
            # Equalization
            #pmi_maps = exposure.equalize_hist(pmi_maps[:, :, 0])
            # Adaptive Equalization

            pmi_maps = exposure.equalize_adapthist(pmi_maps[:, :, 0], clip_limit=0.03)
    else:
        input_img = image#.transpose(2, 0, 1)
        img_name = 'tmp_img.npy'
        pmi_dir = '/data/resist_data/SimCrack/tmp_pmi/'
        npy_path='/data/resist_data/SimCrack/tmp_pmi/tmp_img.npy'
        input_img = torch.Tensor(input_img)
        save_pmi_maps(input_img, args, pmi_dir, img_name)
        pmi_maps = np.load(npy_path)  # .transpose(1,2,0)
        if args.histequalize_pmi:
            ## Equalization
            #img_eq = exposure.equalize_hist(pmi_maps[:, :, 0])

            # Adaptive Equalization
            pmi_maps = exposure.equalize_adapthist(pmi_maps[0, :, :], clip_limit=0.03)
            pmi_maps = np.expand_dims(pmi_maps,axis=0)
    return pmi_maps

def main(args):
    # Load the image
    print('Starting Eval..',args.arch_name)
    original_image = load_image(args.image_path)
    if args.arch_name == 'unet' or args.arch_name == 'pmiunet':
        model = UNet(args.input_ch, 2)
    elif args.arch_name == 'transunet':
        model = TransUNet(img_dim=256,
                          in_channels=1,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=2)
    elif args.arch_name == 'munet':
        model = MultiUNet(args.input_ch, 2)
    elif args.arch_name == 'munet_pmi':
        model = SegPMIUNet(args.input_ch, 2)
    elif args.arch_name=='att_unet':
        model = AttU_Net(args.input_ch,2)
    elif args.arch_name == 'cons_unet' or args.arch_name == '2unet':
        model = ConsNet(args.input_ch, 2, att=args.att_connection, consistent_features=args.cons_loss,
                        img_size=args.input_size)

    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_path = os.path.join(args.model_path)
    if os.path.isfile(full_path):
        print("=> loading checkpoint '{}'".format(full_path))
        checkpoint = torch.load(full_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(full_path, checkpoint['epoch']))
    # Load the model
    model = model.eval()
    # Move the model to the GPU if available
    model = model.to(device)
    # Get the segmentation map
    if args.patchwise_eval:
        print('Patchwise Evaluation...')
        #image = Image.open(args.image_path)
        #image = ImageOps.exif_transpose(image)
        #input_image = np.asarray(image)

        preprocessed_image = preprocess_image(original_image,skip_resize=True, input_size=args.input_size)
        image = preprocessed_image.to(device)[0]
        #input_image = input_image[:, :, :3]
        # greyscale input
        #input_image = np.dot(input_image[..., :3], [0.299, 0.587, 0.114])
        #image = torch.unsqueeze(torch.from_numpy(input_image), dim=0).float()
        #print('shape second method',image.shape) #torch.Size([ 1, 2048, 2048])

        # Replace 'output_image.jpg' with your desired output file path
        if args.arch_name == 'cons_unet':
            print('Evaluating PMI..')
            image_for_pmi = Image.open(args.image_path)
            image_for_pmi = ImageOps.exif_transpose(image_for_pmi)
            image_for_pmi = np.asarray(image_for_pmi)
            image_pmi = get_pmi_map(image_for_pmi,args, patchwise=args.patchwise_eval)  # data['pmi_map'].to(device)
            print('PMI Evaluation done..',image_pmi.shape)
            image_pmi= torch.unsqueeze(torch.Tensor(image_pmi),0)
        else:
            image_pmi=None
        segmentation_map = evaluate_img_patchwise(image,args.input_size,args.patch_size,model,img_pmi=image_pmi)
    else:
        # Preprocess the image for the model
        preprocessed_image = preprocess_image(original_image,skip_resize=False, input_size=args.input_size)
        image = preprocessed_image.to(device)
        image_from_tensor = transforms.ToPILImage()(image[0])
        # Save the PIL image to a file
        image_from_tensor.save(
            '/data/resist_data/SimCrack/outs/output_image_test2.jpg')
        with torch.no_grad():
            #TODO if the input is a pmi image. Work on it here
            #output = model(preprocessed_image)['out']
            if args.arch_name == 'munet':
                _, output, _ = model(image)
            elif args.arch_name == 'munet_pmi':
                output, _ = model(image)
            elif args.arch_name == 'cons_unet':
                image_for_pmi = Image.open(args.image_path)
                image_for_pmi = ImageOps.exif_transpose(image_for_pmi)
                image_for_pmi = image_for_pmi.resize((args.input_size, args.input_size))
                image_for_pmi = np.asarray(image_for_pmi)
                pmi_map = get_pmi_map(image_for_pmi,args,patchwise=args.patchwise_eval)#data['pmi_map'].to(device)
                pmi_map = torch.unsqueeze(torch.Tensor(pmi_map), 0)
                output, output_pmi, _ = model(image, pmi_map)
            elif args.arch_name == '2unet':
                output, output_pmi, _ = model(image, image)
            else:
                output = model(image)
        # Postprocess the output to get the segmentation map
        segmentation_map = torch.argmax(output, dim=1).float().detach().cpu().numpy()
        #prediction = torch.argmax(output, dim=1).float().float().detach().cpu().numpy()
        if (args.arch_name == 'cons_unet' or args.arch_name == '2unet'):
            segmentation_map_pmi = torch.argmax(output_pmi, dim=1).float().detach().cpu().numpy()
            #prediction_pmi = torch.argmax(output_pmi, dim=1).float().detach().cpu().numpy()
            prediction = np.add(segmentation_map_pmi, segmentation_map)
            segmentation_map = segmentation_map_pmi#prediction > 0.5
        segmentation_map=segmentation_map[0]
    # Save the segmentation map as an image
    cv2.imwrite(args.output_path, segmentation_map * 255)
    #segmentation_image = Image.fromarray(np.uint8(segmentation_map)) #
    #segmentation_image.save(args.output_path)
    #/data/resist_data/SimCrack/workspace/trained_nets/cons_unet/SimResist/2023-01-11_19-40-37
if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'False'
    parser = argparse.ArgumentParser(description="Evaluate Semantic Segmentation on an Image")
    parser.add_argument("--image_path", type=str,default='/data/resist_data/SimCrack/test_imgs/strymonas_0000512_3000_844.png', help="Path to the input image")
    parser.add_argument("--output_path", type=str,default='/data/resist_data/SimCrack/outs/strymonas_0000512_3000_844.png', help="Path to save the segmentation map image")
    parser.add_argument("--patchwise_eval", type=bool,default=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str,default='/data/resist_data/SimCrack/workspace/trained_nets/cons_unet/SimResist/2023-01-23_14-59-39/best_model.pth.tar', help="Path to the input image")
    parser.add_argument("--arch_name", type=str,default='cons_unet', help="Path to the input image")
    parser.add_argument("--input_ch", type=int,default=1, help="Path to the input image")
    #/data/resist_data/SimCrack/workspace/trained_nets/cons_unet/SimResist/2023-01-23_14-59-39/best_model.pth.tar , /data/resist_data/SimCrack/pretrained_models/unet.pth.tar
    parser.add_argument("--phi_value", type=float,default=1.25, help="Path to the input image")
    parser.add_argument("--histequalize_pmi", type=bool,default=True, help="Path to the input image")
    parser.add_argument("--neighbour_size", type=int,default=5, help="Path to the input image")
    parser.add_argument("--patch_size", type=float,default=512, help="Path to the input image")
    parser.add_argument("--input_size", type=float,default=256, help="Path to the input image")
    parser.add_argument("--att_connection",type=bool,default=False, help="Path to the input image")
    parser.add_argument("--cons_loss",type=bool,default=True, help="Path to the input image")

    args = parser.parse_args()

    main(args)
