import cv2
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
import xml.etree.ElementTree as ElementTree
import torch
import torchvision
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
import random
import scipy


class Intensity():
    def __call__(self, x):
        return x.sum(axis=2) // 3


class Saturation():
    def __call__(self, x):
        i = x.sum(axis=2)
        min_c = (np.minimum(np.minimum(x[:, :, 0], x[:, :, 1]), x[:, :, 2]))
        return 1 - np.divide(min_c, i, out=np.zeros(min_c.shape), where=i != 0)


class GaussianNoise():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        return np.clip(x + np.random.normal(0, self.sigma, x.shape), 0, 255).astype(np.uint8)


class ResizeNP_smallside():
    def __init__(self, size, list=False):
        self.size = size

    def __call__(self, img):
        if img.shape[0] < img.shape[1]:
            size_x = self.size
            size_y = int(img.shape[1] * (size_x / img.shape[0]))
        else:
            size_y = self.size
            size_x = int(img.shape[0] * (size_y / img.shape[1]))
        return cv2.resize(img, dsize=(size_y, size_x), interpolation=cv2.INTER_LINEAR)


class PIL2NP():
    def __call__(self, img):
        return np.asarray(img)


class ResizeNP():
    def __init__(self, size_x, size_y, list=False):
        self.size_x = size_x
        self.size_y = size_y

    def __call__(self, img):
        img = cv2.resize(img, dsize=(self.size_y, self.size_x), interpolation=cv2.INTER_LINEAR)
        return img


class RandomRotateNP():
    def __init__(self, degree):
        self.d = degree

    def __call__(self, img):
        d = random.uniform(0, self.d)
        return scipy.ndimage.rotate(img, d, reshape=False)


class CenterCropNP():
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

    def __call__(self, img):
        x1 = max(0, int(img.shape[1] / 2 - self.size_x / 2))
        x2 = min(int(img.shape[1] / 2 + self.size_x / 2), img.shape[1])
        y1 = max(0, int(img.shape[0] / 2 - self.size_y / 2))
        y2 = min(int(img.shape[0] / 2 + self.size_y / 2), img.shape[0])
        return img[y1:y2, x1:x2]


class RandomHorizontalFlipNP():
    def __call__(self, img):
        if random.randint(0, 1):
            return cv2.flip(img, 1)
        return img


class RandomCropNP():
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

    def __call__(self, img):
        start_x = random.randint(0, (img.shape[1] - self.size_x))
        start_y = random.randint(0, (img.shape[0] - self.size_y))
        return img[start_y:start_y + self.size_y, start_x:start_x + self.size_x, :]


class ToFloatTensor():
    def __call__(self, x):
        return x.float()


class CropRoi_extra5():
    def __call__(self, x):
        img, (x1, y1, x2, y2) = x
        width, height = x2 - x1, y2 - y1
        '''#add 10 %
        x1 = min(0,x1 - 0.05 * width)
        x2 = max(img.size[0], x2 + 0.05 * width)
        y1 = min(0,y1 - 0.05 * height)
        y2 = max(img.size[1], y2 + 0.05 * height)
        '''
        img, (x1, y1, x2, y2) = x
        width, height = x2 - x1, y2 - y1
        # add 10 %
        x1 = min(0, int(x1 - 0.05 * width))
        x2 = max(img.shape[0], int(x2 + 0.05 * width))
        y1 = min(0, int(y1 - 0.05 * height))
        y2 = max(img.shape[1], int(y2 + 0.05 * height))
        # print(x1,x2,y1,y2)
        # return img.crop((x1,y1,x2,y2))
        return img[y1:y2, x1:x2]


class CropRoi():
    def __call__(self, x):
        img, (x1, y1, x2, y2) = x
        # return img.crop((x1,y1,x2,y2))
        return img[y1:y2, x1:x2]


class Histogram_Equalization_HSVNP():

    def __call__(self, img):
        # img = np.array(img)
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)
        # img = Image.fromarray((img*255).astype(np.uint8))
        # opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow('hsv_equalized',opencvImage)
        # cv2.waitKey(3000)
        return img

    def __str__(self):
        return "Histogram Equalization(based on HSV)"


# from skimage import color, exposure, transform

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def load_image_np(path):
    img = Image.open(path)
    img = img.convert('RGB')
    return np.asarray(img)


NO_ROI = 0
APPLY_ROI = 1
RETURN_ROI = 2


class Roi_image_loader():
    def __init__(self, path, roi_mode=NO_ROI):
        self.rois = {}
        self.roi_mode = roi_mode
        if os.path.isdir(path):
            # if (path).is_dir():
            root = path
            classes = [d.name for d in os.scandir(root) if d.is_dir()]
            for c in classes:
                csv_path = root + c + "/GT-" + c + ".csv"
                self.read_rois_from_csv(root + c + "/", csv_path)
        elif path.endswith(".csv"):
            splited_path = path.split('/')
            root = path[:-len(splited_path[-1])]
            self.read_rois_from_csv(root, path)

    def read_rois_from_csv(self, root, path, verbose=False):
        data = pd.read_csv(path, sep=";")
        for i in range(data.shape[0]):
            filename = data['Filename'][i]
            rois = (data['Roi.X1'][i], data['Roi.Y1'][i], data['Roi.X2'][i], data['Roi.Y2'][i])
            self.rois.update({root + filename: rois})
            if verbose:
                print(root + filename)

    def __call__(self, path):
        with open(path, 'rb') as f:
            img = load_image_np(f)
            if self.roi_mode == NO_ROI:
                return img
            crop_area = self.rois[path]
            if self.roi_mode == APPLY_ROI:
                return img[crop_area]
            return img, crop_area


RED_c = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]  # Circle with red margin
RED_cf = [14, 17]  # Filled Red
BLUE_cf = [33, 34, 35, 36, 37, 38, 39, 40]  # Blue filled circle
RED_t = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]  # Triangle Red Surrounding upword
RED_t2 = [13]  # Triangle red(vorfahrt)
GRAY = [6, 32, 41, 42]  # Gray
YELLOW = [12]  # yellow rectangle(vorfahrt)


class SortedByColoredShape():
    def __init__(self):
        self.map = self.create_map()
        self.num_classes = 43
        self.class_names = ["Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
                            "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
                            "End of speed limit (80km/h)", "Speed limit (100km/h) ", "Speed limit (120km/h)",
                            "No passing", "No passing veh over 3.5 tons", "Right-of-way at intersection",
                            "Priority road", "Yield", "Stop", "No vehicles", "Veh > 3.5 tons prohibited ",
                            "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
                            "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
                            "Traffic signals", "Pedestrians", "Children crossing ", "Bicycles crossing",
                            "Beware of ice/snow", "Wild animals crossing ", "End speed + passing limits ",
                            "Turn right ahead ", "Turn left ahead ", "Ahead only", "Go straight or right",
                            "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
                            "End of no passing ", "End no passing veh > 3.5 tons"]
        inv_map = {v: k for k, v in self.map.items()}
        self.class_names = [self.class_names[i[1]] for i in inv_map.items()]

    def create_map(self):
        idx = 0
        m = {}
        for group in [RED_c, RED_cf, BLUE_cf, RED_t, RED_t2, GRAY, YELLOW]:
            for i in group:
                m.update({i: idx})
                idx += 1
        return m

    def __call__(self, x):
        return self.map[x]


class DatasetFolderTest(DatasetFolder):
    def __init__(self, *args):
        super(DatasetFolderTest, self).__init__(*args)

    def __getitem__(self, index):
        print("getitem %i" % (index))
        result = super().__getitem__(index)
        print("finished getitem %i" % (index))
        return result


class ColorShapeClasses():
    def __init__(self):
        self.map = self.create_map()
        self.num_classes = 7
        self.class_names = ["Circle with red margin", "Filled Red", "Blue filled circle",
                            "Triangle Red Surrounding upword",
                            "Triangle red(Yield)", "Gray ", "yellow rectangle(priority road)"]

    def create_map(self):
        idx = 0
        m = {}
        for group in [RED_c, RED_cf, BLUE_cf, RED_t, RED_t2, GRAY, YELLOW]:
            for i in group:
                m.update({i: idx})
            idx += 1
        return m

    def __call__(self, x):
        return self.map[x]


class KeepInputandRoi():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return np.concatenate([self.transform((x[0], x[1])), x[0]], axis=2), x[1]


class SplitTransform():
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return (self.transform1(x[0]), self.transform2(x[1]))


class ApplyOnBackChannels():
    def __init__(self, transform, channels):
        self.transform = transform
        self.channels = channels

    def __call__(self, x):
        return np.concatenate([x[:, :, :-self.channels], self.transform(x[:, :, -self.channels:])], axis=2)


class MyDatasetFolder(DatasetFolder):
    def __init__(self, *args, decompositions=[], inputAdditionalTagret=False):
        self.decompositions = decompositions
        self.toTensor = transforms.Compose([transforms.ToTensor(), ToFloatTensor()])
        self.inputAdditionalTagret = inputAdditionalTagret
        super().__init__(*args)

    def __getitem__(self, index):
        original_sample, label = super().__getitem__(index)
        sample = original_sample
        if len(self.decompositions) > 0:
            transformed_sample = []
            for op in self.decompositions:
                transformed_sample.append(op(sample))
            sample = np.concatenate(transformed_sample, axis=2)
        if self.inputAdditionalTagret:
            return self.toTensor(sample), (label, self.toTensor(original_sample))
        else:
            return self.toTensor(sample), label


class DatasetwithCSV(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, target_transform=None, roi_mode=NO_ROI,
                 decompositions=[], inputAdditionalTagret=False):
        self.annotations = pd.read_csv(csv_file, sep=";")
        self.Roi_img_loader = Roi_image_loader(root_dir + "GT-final_test.test.csv", roi_mode)
        self.roi_mode = roi_mode
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.toTensor = transforms.Compose([transforms.ToTensor(), ToFloatTensor()])
        self.decompositions = decompositions
        self.inputAdditionalTagret = inputAdditionalTagret

    def __getitem__(self, i):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[i, 0])
        image = self.Roi_img_loader(img_path)
        label = self.annotations.iloc[i, 7]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = image
        if len(self.decompositions) > 0:
            transformed_sample = []
            for op in self.decompositions:
                transformed_sample.append(op(sample))
            sample = np.concatenate(transformed_sample, axis=2)
        if self.inputAdditionalTagret:
            return self.toTensor(sample), (label, self.toTensor(image))
        else:
            return self.toTensor(sample), label

    def __len__(self):
        return len(self.annotations)


class CODEBRIMSplit(torchvision.datasets.ImageFolder):
    """
    definition of class for reading data-split images and class labels, and iterating
    over the datapoints
    Parameters:
        root (string): directory path for the data split
        xml_list (list): list of paths to xmls for defect and background meta-data
        transform (torchvision.transforms.Compose): transforms for the input data
        target_transform (callable): transform for the targets
        loader (callable): for loading an image given its path
    Attributes:
        file_list (dictionary): dictionary of file names (keys) and the corresponding
                                class labels (values)
        num_classes (int): number of classes in the dataset (6)
    """

    def __init__(self, root, xml_list, transform=None, target_transform=None,
                 loader=torchvision.datasets.folder.default_loader):
        super(CODEBRIMSplit, self).__init__(root, transform, target_transform, loader)
        self.file_list = {}
        self.num_classes = 6
        for i, file_name in enumerate(xml_list):
            last_dot_idx = file_name.rfind('.')
            f_name_idx = file_name.rfind('/')
            root_path = file_name[f_name_idx + 1: last_dot_idx]
            tree = ElementTree.parse(file_name)
            root = tree.getroot()
            for defect in root:
                crop_name = list(defect.attrib.values())[0]
                target = self.compute_target_multi_target(defect)
                self.file_list[os.path.join(root_path, crop_name)] = target

    def __getitem__(self, idx):
        """
        defines the iterator for the dataset and returns datapoints in the form of tuples
        Parameters:
            idx (int): index to return the datapoint from
        Returns:
            a datapoint tuple (sample, target) for the index
        """
        image_batch = super(CODEBRIMSplit, self).__getitem__(idx)[0]
        image_name = self.imgs[idx][0]
        f_name_idx = image_name.rfind('/')
        f_dir_idx = image_name[: f_name_idx].rfind('/')
        de_lim = image_name.rfind('_-_')
        file_type = image_name.rfind('.')
        if de_lim != -1:
            name = image_name[f_dir_idx + 1: de_lim] + image_name[file_type:]
        else:
            name = image_name[f_dir_idx + 1:]
        return [image_batch, self.file_list[name]]

    def compute_target_multi_target(self, defect):
        """
        enumerates the class-label by defining a float32 numpy array
        Parameters:
            defect (string): the class labels in the form of a string
        Returns:
            the enumerated version of the labels in the form of a numpy array
        """
        out = np.zeros(self.num_classes, dtype=np.float32)
        for i in range(self.num_classes):
            if defect[i].text == '1':
                out[i] = 1.0
        return out


class Tensor_Dataset_From_PIL(Dataset):
    def __init__(self, inputs, targets, dynamic_transforms, multi_modal):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputs = inputs
        self.targets = targets
        self.dynamic_transforms = dynamic_transforms
        self.multi_modal = multi_modal

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        next_input = self.inputs[idx]
        next_target = self.targets[idx]

        next_input = Image.fromarray(np.uint8(next_input * 255))

        if self.multi_modal:
            # vanilla_img, rg_img, lbp_img, wavelet_img
            vanilla_img = self.dynamic_transforms[0](next_input)
            rg_img = self.dynamic_transforms[1](next_input)
            lbp_img = self.dynamic_transforms[2](next_input)
            wavelet_img = self.dynamic_transforms[3](next_input)

            return vanilla_img, rg_img, lbp_img, wavelet_img, next_target

        else:
            if self.dynamic_transforms:
                next_input = self.dynamic_transforms(next_input)

            return next_input, next_target


class CODEBRIM():
    """
    definition of CODEBRIM dataset, train/val/test splits, train/val/test loaders
    Parameters:
        args (argparse.Namespace): parsed command line arguments
        is_gpu (bool): if computational device is gpu or cpu
    Attributes:
        num_classes (int): number of classes in the dataset (= 6)
        dataset_path (string): path to dataset folder
        dataset_xml_list (list): list to dataset meta-data
        train_set (CODEBRIMSplit): train split
        val_set (CODEBRIMSplit): validation split
        test_set (CODEBRIMSplit): test split
        train_loader (torch.utils.data.DataLoader): data-loader for train-split
        val_loader (torch.utils.data.DataLoader): data-loader for val-split
        test_loader (torch.utils.data.DataLoader): data-loader for test-split
    """

    def __init__(self, batch_size, operators=[], save_disk=False, p=1, contrastive=False, train_transfo=None,
                 test_transfo=None, multi_modal=False):
        self.num_classes = 6
        self.batch_size = batch_size
        self.patch_size = 224  # default 224
        self.operators = operators
        self.save_disk = save_disk
        self.target_transforms = None
        self.contrastive = contrastive
        self.multi_modal = multi_modal

        self.train_transfo = train_transfo
        self.test_transfo = test_transfo
        self.name = "CODEBRIM(batch_size=%i, patch_size =%i)" % (self.batch_size, self.patch_size)

        self.dataset_path = "/data/resist_data/datasets/CODEBRIM/classification_dataset_balanced"  # "../../data/CODEBRIM/classification_dataset_balanced"
        self.dataset_xml_list = [os.path.join(self.dataset_path, 'metadata/background.xml'),
                                 os.path.join(self.dataset_path, 'metadata/defects.xml')]
        self.trainset, self.valset, self.testset, self.viz_images = self.get_dataset(self.patch_size, p)  #
        self.trainloader, self.valloader, self.testloader = self.get_dataset_loader(self.batch_size, 4)  #
        print("Created Trainset(%i), Validationset(%i), Testset(%i)" % (
            len(self.trainset), len(self.valset), len(self.testset)))

    def createandload_Dataset(self, root, static_transforms, save_disk, p=1, test_set=False):
        if save_disk:
            key = ""
            for t in static_transforms.transforms:
                key += class_info(t)
            # key = str(static_transforms.transforms)
            key = os.path.join(root, key)
            if (not os.path.isfile(key)):
                print("save data to : ", key)
                d = CODEBRIMSplit(root, self.dataset_xml_list, transform=static_transforms)
                inputs, targets = [], []
                print(len(d))
                for i, (inp, target) in tqdm(enumerate(d)):
                    inputs.append(inp)
                    targets.append(target)
                    if i > 500:
                        break

                torch.save((inputs, targets), key)
            else:
                print("load data from", key)
                inputs, targets = torch.load(key)

        else:
            d = CODEBRIMSplit(root, self.dataset_xml_list, transform=static_transforms)
            print(len(d))
            inputs, targets = [], []
            for i, (inp, target) in tqdm(enumerate(d)):
                inputs.append(inp)
                targets.append(target)
                if i > 500:
                   break

        if p != 1:
            c = random.sample(range(len(inputs)), int(p * len(inputs)))
            inputs = [inputs[i] for i in c]
            targets = [targets[i] for i in c]
        if not test_set:
            return Tensor_Dataset_From_PIL(inputs, targets, self.train_transfo, self.multi_modal)
        else:
            return Tensor_Dataset_From_PIL(inputs, targets, self.test_transfo, self.multi_modal)

    def get_dataset(self, patch_size, p):
        """
        return dataset splits
        Parameters:
            patch_size (int): patch-size to rescale the images to

        Returns:
            train_set, val_set, test_set of type lib.Datasets.datasets.CODEBRIMSplit
        """
        print(self.operators)
        transform_train_static = transforms.Compose([PIL2NP(), ResizeNP_smallside(patch_size)] + self.operators)

        train_set = self.createandload_Dataset(os.path.join(self.dataset_path, 'train'),
                                               transform_train_static,
                                               self.save_disk, p)
        val_set = self.createandload_Dataset(os.path.join(self.dataset_path, 'val'),
                                             transform_train_static,
                                             self.save_disk)
        test_set = self.createandload_Dataset(os.path.join(self.dataset_path, 'test'),
                                              transform_train_static,
                                              self.save_disk)

        viz_images = []
        dataset = CODEBRIMSplit(os.path.join(self.dataset_path, 'train'), self.dataset_xml_list)
        for p in [0.02, 0.1, 0.18, 0.25, 0.3, 0.35, 0.4, 0.45, 0.52, 0.58, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95]:
            img, _ = dataset[int(p * len(dataset))]
            t1 = transforms.Compose([PIL2NP(), ResizeNP_smallside(patch_size), CenterCropNP(patch_size, patch_size)])
            t_op = transforms.Compose(self.operators)
            t2 = transforms.Compose([transforms.ToTensor(), ToFloatTensor()])
            img = t1(img)
            img_t = t_op(img)
            viz_images.append((t2(img), t2(img_t)))

        return train_set, val_set, test_set, viz_images  #

    def get_dataset_loader(self, batch_size, workers):
        """
        defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): mini batch size in data loader
            workers (int): number of parallel cpu threads for data loading
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
            train_loader, val_loader, test_loader of type torch.utils.data.DataLoader
        """
        train_loader = torch.utils.data.DataLoader(self.trainset, num_workers=workers, batch_size=batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.valset, num_workers=workers, batch_size=batch_size,
                                                 shuffle=False)
        test_loader = torch.utils.data.DataLoader(self.testset, num_workers=workers, batch_size=batch_size,
                                                  shuffle=False)

        return train_loader, val_loader, test_loader  #

def get_codebrim_dataloader(imsize=(512,512)):
    transform_train = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])
    transform_valid = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])

    c_dataset = CODEBRIM(4, train_transfo=transform_train, test_transfo=transform_valid)

    train_dataset = c_dataset.trainset
    test_dataset = c_dataset.testset
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=16,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=16
    )

    return (
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )


if __name__ == "__main__":

    train_loader, _, test_loader, _ = get_codebrim_dataloader()
    import matplotlib.pyplot as plt

    for idx, (img, label) in enumerate(test_loader):
        print(img.shape)

        img = img[0]
        print(img.shape)
        plt.Figure()

        plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())

