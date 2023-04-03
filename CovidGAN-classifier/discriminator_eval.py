import torch
from torch import nn
from torch.nn import Sequential, BCELoss

import os 
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import random
from PIL import Image
import pickle
import subprocess
import hashlib
from datetime import datetime
from sklearn.metrics  import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score
import neptune 
#from torch.cuda import is_cuda_enabled
#from torch.nn import RNN
#from torch.autograd import Variable
DATA_ROOT_DIR = 'CovidData/Lung_Segmentation_Data'

DEFAULT_LOSS = torch.nn.BCELoss()
LOSS_FN = torch.nn.CrossEntropyLoss()
BATCH_SIZE = 128

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#now = datetime.now()
#hash_id = hashlib.md5(now.strftime("%Y-%m-%d_%H_%M_%S").encode("utf-8")).hexdigest() 
#NEPTUNE_ID = hash_id

#torch.set_num_threads(8)

gan_directories = {
    'Test_orig_0.8' : {'dir':'2022-12-14_12-57-44','best':'127.0.0.1-5002'},
    'Test_orig_0.6' : {'dir':'2022-12-14_15-04-49','best':'127.0.0.1-5001'},
    'Test_orig_0.4' : {'dir':'2022-12-14_16-48-34','best':'127.0.0.1-5002'},
    'Test_orig_0.2' : {'dir': '2022-12-14_18-54-13','best':'127.0.0.1-5002'},
    
    'Test_0_0.8' : {'dir': '2022-12-15_09-10-08','best':''},
    'Test_0_0.6' : {'dir':'2022-12-15_10-41-13','best':''},
    'Test_0_0.4' : {'dir':'2022-12-15_11-34-52','best':''},
    'Test_0_0.2' : {'dir':'2022-12-15_12-31-45','best':''},

    'Test_1_0.8' : {'dir':'2022-12-15_12-57-15','best':'127.0.0.1-5003'},
    'Test_1_0.6' : {'dir':'2022-12-15_14-07-49','best':'127.0.0.1-5001'},
    'Test_1_0.4' : {'dir':'2022-12-15_15-01-39','best':'127.0.0.1-5000'},
    'Test_1_0.2' : {'dir':'2022-12-15_15-41-37','best':'127.0.0.1:5002'},

    'Test_2_0.8' : {'dir':'2022-12-15_16-08-24','best':'127.0.0.1-5001'},
    'Test_2_0.6' : {'dir':'2022-12-15_17-22-46','best':'127.0.0.1-5001'},
    'Test_2_0.4' : {'dir':'2022-12-15_18-18-53','best':'127.0.0.1-5001'},
    'Test_2_0.2' : {'dir':'2022-12-15_19-04-53','best':'127.0.0.1-5002'},

    'Test_3_0.8' : {'dir': '2022-12-15_19-31-00','best':'127.0.0.1-5001'},
    'Test_3_0.6' : {'dir':'2022-12-15_20-43-34','best':'127.0.0.1-5003'},
    'Test_3_0.4' : {'dir':'2022-12-15_21-39-35','best':'127.0.0.1-5002'},
    'Test_3_0.2' : {'dir':'2022-12-15_22-21-06','best':'127.0.0.1-5000'},
     }

def create_discriminator(split, data_ratio):
    complexity = 64
    net = Sequential(
                nn.Conv2d(1, complexity, 4, 2, 3),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(complexity, complexity * 2, 4, 2, 1),
                nn.BatchNorm2d(complexity * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(complexity * 2, complexity * 4, 4, 2, 1),
                nn.BatchNorm2d(complexity * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(complexity * 4, complexity * 4, 4, 2, 1),
                nn.BatchNorm2d(complexity * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(complexity * 4, 1, 8, 1, 0),
                nn.Sigmoid()
            )
    lippi_dir = '/home/bbernard/lipizzaner-covidgan-master/src/'
    gan_dir = gan_directories[f'Test_{split}_{data_ratio}']['dir'] 
    best_gen = gan_directories[f'Test_{split}_{data_ratio}']['best'] 
    src_file = os.path.join(lippi_dir, f'output/lipizzaner_gan/master/{gan_dir}/{best_gen}/discriminator-{best_gen}.pkl')
    net.load_state_dict(torch.load(src_file))
    return net

class CustomDataset(torchvision.datasets.ImageFolder):
    """
        A simple imagedatset for storing data
    """
    #Imagefolder for efficiency
    def __init__(self, images, transform):
        target_dir = os.path.join(DATA_ROOT_DIR, 'original')
        super().__init__(target_dir, transform)
        self.samples = images
        self.imgs = images


def DatasetMaker(split, data_ratio=1, transform = None, geoaugment=False):
    """
        Returns a CustomDataset with given parameters
        split: str, determines train-test to use; options: 'orig', '0', '1', '2', '3'
        data_ratio (optional): int, the ratio of the training covid data to be used
                    options:
                        '1' : all data
                        '0.8' : 80% of training images
                        '0.6' : 60% of training images
                        '0.4' : 40% of training images
                        '0.2' : 20% of training images
        transform (optional): torch.Compose instance, sets the dataset's transforms
        geougment (optional): bool, uses basic data augmentation techmiques
        seed (optional): int, seed to use for reproducibility
    """
    #This code prepares my fixed set of images to replace the Imagefolder's original images
    #Maybe this could be done nicer with overwriting the DatasetFolder's find_classes method, but currently this works
    # the idea is to make (route, index) pairs for later loading in images

    #if is_test_valid(split) is False:
    #    return
    
    classes = ['normal', 'viral', 'covid']

    #Determining the root directories for original images 
    orig_dirs = {
    'normal' : f'{DATA_ROOT_DIR}/original/Normal',
    'viral' : f'{DATA_ROOT_DIR}/original/Non-Covid',
    'covid' : f'{DATA_ROOT_DIR}/original/COVID-19'
    }
    #Determining the root directories for files which contain the names of the COVID19 pictures 
    indicies_files = {
        'gan_0.8' : f'{DATA_ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_0.8_gan.pkl', 
        'gan_0.6' : f'{DATA_ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_0.6_gan.pkl',
        'gan_0.4' : f'{DATA_ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_0.4_gan.pkl',
        'gan_0.2' : f'{DATA_ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_0.2_gan.pkl',
        'test'  : f'{DATA_ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_test.pkl',
        'train'  : f'{DATA_ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_train_and_val.pkl'
    }
    class_idx = {
        'covid': 1, 
        'viral': 0,
        'normal': 0,
        'gan_0.8' : 1,
        'gan_0.6' : 1,
        'gan_0.4' : 1,
        'gan_0.2' : 1,
        'non-covid' : 0
    }
    idx_to_class ={
        0:'non-covid', #'covid',
        1:'covid',#'viral',
        2:'non-covid' #'normal'
    }

    #The dictionary that will contain the route for the several image classes
    source_dir = {}
    train_images = []

    #creating the sources for the classes
    for class_name in classes: 
        source_dir[class_name] = orig_dirs[class_name]
    
    #Get all training images
    file = indicies_files['train'] 
    imgs = load_images_from_file(file)
    imgs = [*imgs[0],*imgs[1]] #The file contains (train, val) sets
    
    #Making the paths for the training images
    for x in imgs: 
        class_of_x = idx_to_class[x[1]] #The images are saved in (image_name, class_index) format
        item = os.path.join(source_dir[class_of_x],x[0]),class_idx[class_of_x] 
        train_images.append(item)
    #If data_ratio is not 1, we need to change the covid images of the dataset
    if data_ratio!=1:
        num_of_covid_imgs = 0
        train_images = [x for x in train_images if x[1]!=class_idx['covid']]
        file = indicies_files[f'gan_{data_ratio}']
        imgs = load_images_from_file(file)
        for x in imgs:
            class_of_x = idx_to_class[x[1]]
            item = os.path.join(source_dir[class_of_x],x[0]),class_idx[class_of_x] 
            train_images.append(item)
    
    #Making the paths for the test images
    test_images = []
    test_imgs = load_images_from_file(indicies_files['test'])
    for x in test_imgs:
        class_of_x = idx_to_class[x[1]]
        item = os.path.join(source_dir[class_of_x],x[0]),class_idx[class_of_x] #This should be correct
        test_images.append(item)

    #Making some basic transforms 
    if transform is None:
        transforms = [ torchvision.transforms.Resize(size=(128, 128)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    torchvision.transforms.Grayscale(num_output_channels=1)]
    else:
        transforms = transform

    #Checking wether geometric augmentation is needed ( classic augmentation methods)
    if geoaugment:
        augmentation_transforms = [#torchvision.transforms.RandomHorizontalFlip(), #(should be useful, causes confusion with gans)
                                torchvision.transforms.RandomAffine(4)]
        transforms = [augmentation_transforms + transforms]                      
    transforms = torchvision.transforms.Compose(transforms)
    train_dataset = CustomDataset(train_images, transforms)
    #val_dataset = CustomDataSet(val_images, transforms)
    test_dataset = CustomDataset(test_images,  transforms)
    all_dataset = CustomDataset([*train_images, *test_images],  transforms)
    return train_dataset, test_dataset, all_dataset

def load_images_from_file(file):
    with open(file, 'rb') as file:
        data = pickle.load(file)
    return data   

def test(net, loss_fn, test_dataset, batch_size, shuffle):
    test_dl =  torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=0, shuffle= shuffle)
    val_loss = 0.
    val_accuracy = 0.

    net.eval()
    
    val_batch_num = 0
    val_iter = iter(test_dl)
    y_true = []
    y_pred = []
    while val_batch_num < len(val_iter):
        images, labels = next(val_iter)
        y_true.extend(labels)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = net(images)
        loss = loss_fn(outputs, labels)
        val_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.detach().cpu())
        val_accuracy += sum((preds == labels).cpu().numpy())
        val_batch_num += 1

    val_loss /= len(test_dl)
    val_accuracy = val_accuracy/len(test_dataset)

    run = neptune.init_run()
    run['algorithm'] = "GAN_eval"

    curr_conf_matrix = confusion_matrix(y_true, y_pred) 
    curr_conf_matrix = curr_conf_matrix / np.sum(curr_conf_matrix)
    im = ConfusionMatrixDisplay(curr_conf_matrix, display_labels=["fake", "real"]).plot()
    run[f'metrics/conf_matrix'].append(im.figure_) #, description=f"Confusion matrix in the iteration: {iteration}"  File.as_image(curr_conf_matrix))
    run[f'metrics/acc'].append( accuracy_score(y_true, y_pred))
    run[f'metrics/bal_acc'].append(balanced_accuracy_score(y_true, y_pred))
    run[f'metrics/recall'].append(recall_score(y_true, y_pred))
    run[f'metrics/precision'].append( precision_score(y_true, y_pred))
    run[f'train/metrics/f1'].append( f1_score(y_true, y_pred))

    run.stop()
    #print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

if __name__=='__main__':
    splits = ['orig', '0', '1', '2', '3']
    ratios = [0.8, 0.6, 0.4, 0.2]
    for split in splits:
        for data_ratio in ratios:
            net = create_discriminator(split, data_ratio)
            _,_,test_dataset = DatasetMaker(split, data_ratio)
            test(net, DEFAULT_LOSS,test_dataset,BATCH_SIZE,True )
            print(f"Finished {split} {data_ratio} run")