import os 
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import random
from PIL import Image
import pickle
import subprocess
import time

ROOT_DIR = 'CovidData/Lung_Segmentation_Data'

torch.set_num_threads(8)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#print(device)

LOSS_FN = torch.nn.CrossEntropyLoss()
#print(device)

#The directories where the gan's pkl is stored for generation
gan_directories = {
    'Test_orig_0.8' : '2022-12-14_12-57-44',
    'Test_orig_0.6' : '2022-12-14_15-04-49',
    'Test_orig_0.4' : '2022-12-14_16-48-34',
    'Test_orig_0.2' : '2022-12-14_18-54-13',
    
    'Test_0_0.8' : '2022-12-15_09-10-08',
    'Test_0_0.6' : '2022-12-15_10-41-13',
    'Test_0_0.4' : '2022-12-15_11-34-52',
    'Test_0_0.2' : '2022-12-15_12-31-45',

    'Test_1_0.8' : '2022-12-15_12-57-15',
    'Test_1_0.6' : '2022-12-15_14-07-49',
    'Test_1_0.4' : '2022-12-15_15-01-39',
    'Test_1_0.2' : '2022-12-15_15-41-37',

    'Test_2_0.8' : '2022-12-15_16-08-24',
    'Test_2_0.6' : '2022-12-15_17-22-46',
    'Test_2_0.4' : '2022-12-15_18-18-53',
    'Test_2_0.2' : '2022-12-15_19-04-53',

    'Test_3_0.8' : '2022-12-15_19-31-00',
    'Test_3_0.6' : '2022-12-15_20-43-34',
    'Test_3_0.4' : '2022-12-15_21-39-35',
    'Test_3_0.2' : '2022-12-15_22-21-06',
     }

class CustomDataset(torchvision.datasets.ImageFolder):
    """
        A simple imagedatset for storing data
    """
    #Imagefolder for efficiency
    def __init__(self, images, transform):
        target_dir = os.path.join(ROOT_DIR, 'original')
        super().__init__(target_dir, transform)
        self.samples = images
        self.imgs = images


def DatasetMaker(split, mode=None, data_ratio=1, transform = None, geoaugment=False, seed = 0):
    """
        Returns a CustomDataset with given parameters
        split: str, determines train-test to use; options: 'orig', '0', '1', '2', '3'
        mode: str, 'oversampling', 'gan' or None
            'oversampling' : oversample with real images to balance classes
                    'gan' : balance datasets with gan generated images (uses data_ratio to figure out which gan to use)
                    None  : dataset won't be balanced
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

    if is_test_valid(split) is False:
        return
    
    classes = ['normal', 'viral', 'covid']

    #Determining the root directories for original images 
    orig_dirs = {
    'normal' : f'{ROOT_DIR}/original/Normal',
    'viral' : f'{ROOT_DIR}/original/Non-Covid',
    'covid' : f'{ROOT_DIR}/original/COVID-19'
    }
    #Determining the root directories for generated images 
    fake_dirs = {
        'gan_0.8' : f'{ROOT_DIR}/generated/Test_{split}/gan_0.8',
        'gan_0.6' : f'{ROOT_DIR}/generated/Test_{split}/gan_0.6',
        'gan_0.4' : f'{ROOT_DIR}/generated/Test_{split}/gan_0.4',
        'gan_0.2' : f'{ROOT_DIR}/generated/Test_{split}/gan_0.2'
    }
    #Determining the root directories for files which contain the names of the COVID19 pictures 
    indicies_files = {
        'gan_0.8' : f'{ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_0.8_gan.pkl', 
        'gan_0.6' : f'{ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_0.6_gan.pkl',
        'gan_0.4' : f'{ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_0.4_gan.pkl',
        'gan_0.2' : f'{ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_0.2_gan.pkl',
        'test'  : f'{ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_test.pkl',
        'train'  : f'{ROOT_DIR}/Indicies_files/Test_{split}/{split}_split_train_and_val.pkl'
    }
    class_idx = {
        'covid': 0, 
        'viral': 1,
        'normal': 2,
        'gan_0.8' : 0,
        'gan_0.6' : 0,
        'gan_0.4' : 0,
        'gan_0.2' : 0
    }
    idx_to_class ={
        0: 'covid',
        1: 'viral',
        2: 'normal'
    }
    valid_ratios = [1, 0.8, 0.6, 0.4, 0.2]

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
    num_of_covid_imgs = 0 #?
    for x in imgs: 
        class_of_x = idx_to_class[x[1]] #The images are saved in (image_name, class_index) format
        item = os.path.join(source_dir[class_of_x],x[0]),class_idx[class_of_x] 
        if item[1] == class_idx['covid']: num_of_covid_imgs +=1 #?
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
            if item[1] == class_idx['covid']: num_of_covid_imgs +=1 
            train_images.append(item)
    
    #If picture generation is needed we want to know the average class size and missing number of covid images
    average_class_size = round((len(train_images)-num_of_covid_imgs)/2)
    missing_images = max(0, average_class_size - num_of_covid_imgs)

    #Determining wether image generation is needed and which kind of it then making the generation
    if mode=='gan' and (data_ratio in valid_ratios):
        gan = f'gan_{data_ratio}'
        gan_dir = fake_dirs[gan]
        generate_images_to_dir(split, data_ratio, gan, gan_dir, missing_images) #?
        gan_ims = []
        for x in os.listdir(gan_dir): #optimize further
            if x.lower().endswith('jpg'):
                item = os.path.join(gan_dir, x), class_idx['covid']
                gan_ims.append(item)
        sample = random.sample(gan_ims, missing_images)
        train_images = [*train_images,*sample]
    elif mode=='oversampling':
        covid_images = [x for x in train_images if x[1]==class_idx['covid']]
        batch_size = len(covid_images)
        while batch_size <= missing_images:
            train_images = [*train_images, *covid_images]
            missing_images -= batch_size
        if missing_images>0:
            sample = random.sample(covid_images, missing_images)
            train_images = [*train_images, *sample]
    
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
    return train_dataset, test_dataset #val_dataset

def load_images_from_file(file):
    with open(file, 'rb') as file:
        data = pickle.load(file)
    return data   
    
def generate_images_to_dir(split, data_ratio, gan, directory, size):
    """
        Generates pictures with a given gan, to a given directory
    """
    #Goes into Lipizzaner's directory and then generates images with a given GAN (this function is a Lipizzaner built-in method)
    #Then returns into this directory
    
    curr_dir = os.getcwd()
    print(curr_dir)
    
    lippi_dir = '/home/bbernard/lipizzaner-covidgan-master/src/'  #Change this on server
    
    output_dir = os.path.join(curr_dir, directory) #?
   
   #Gan to use is determined by the split and the data_ratio parameters
    gan_dir = gan_directories[f'Test_{split}_{data_ratio}'] 
    src_dir = os.path.join(lippi_dir, f'output/lipizzaner_gan/master/{gan_dir}/127.0.0.1-5000')
    
    config_file = os.path.join(lippi_dir, f'configuration/covid-qu-conv/Test_{split}/covidqu_{data_ratio}.yml')
    
    #man = os.path.join(lippi_dir, 'main.py')

    code =f'python main.py generate --mixture-source {src_dir} -o {output_dir} --sample-size {size} -f {config_file}'
    os.chdir(lippi_dir)
    subprocess.run(code, shell=True)
    os.chdir(curr_dir)


def is_test_valid(test):
    if test in ['orig', '0', '1', '2', '3']: return True
    else: return False

def get_model(name):
    """
        Returns pretrained models
    """
    if name=="resnet":
        resnet18 = torchvision.models.resnet18(pretrained=True)
        resnet18.conv1= torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
        resnet18.get_name = 'resnet18'
        return resnet18
    elif name=="vgg":
        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg16.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        vgg16.classifier[6] = torch.nn.Linear(in_features=4096, out_features=3, bias=True)
        vgg16.get_name = 'vgg16'
        return vgg16
    elif name=="efficient":
        efficient = torchvision.models.efficientnet_b0(pretrained=True)
        efficient.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        efficient.classifier[1] = torch.nn.Linear(in_features=1280, out_features=3, bias=True)
        efficient.get_name = 'efficientnet_b0'
        return efficient
    else:
        print("Not implemented")

def train(epochs, modell, loss_fn, optimizer, train_dataset, test_dataset, batch_size, shuffle ):
    """
        A simple train function 
        Params: 
            epoch: number of epochs to train for
            modell: The neural network to train
            loss_fn: Loss function instance
            optimizer: optimizer instance
            train_dataset (Dataset)
            test_dataset (Dataset)
            batch_size (int)
            shuffle (bool) 
    """
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=0, shuffle= shuffle)
    test_dl =  torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=0, shuffle= shuffle)

    #Logging history
    history = {'train_loss': [],
               'train_accuracy': [],
               'val_loss': [],
               'val_accuracy': []}

    print('Starting training..')
    for e in range(epochs):
        print('='*20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('='*20)

        train_loss = 0.
        train_accuracy = 0.
        train_iter = iter(train_dl)
        
        curr_acc = 0.
        curr_loss = 0.
        sample_num = 0 

        modell.train() # set model to training phase
        
        #Training 
        batch_num = 0
        while batch_num< len(train_dl):
            images, labels = next(train_iter)
            images = images.to(device) 
            labels = labels.to(device) 
            
            outputs = modell(images)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            
            curr_acc += sum((preds == labels).cpu().numpy())
            curr_loss += loss.item()
            sample_num += len(labels)
            print(sample_num)
            
            #At every 20th iteration evaluate the training's state and make an evaluation on a test dataset
            if batch_num%20==0:
                if batch_num != 0:
                  curr_loss/= 20 
                curr_acc/= sample_num
                print(f'Train step: {batch_num} Training Loss: {curr_loss:.4f}, Accuracy: {curr_acc:.4f}')
                history['train_loss'].append(curr_loss)
                history['train_accuracy'].append(curr_acc)
                curr_loss = 0.
                curr_acc = 0.
                sample_num = 0
                
                val_loss = 0.
                val_accuracy = 0.

                modell.eval()
                
                val_batch_num = 0
                val_iter = iter(test_dl)
                
                while val_batch_num < len(val_iter):
                    images, labels = next(val_iter)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = modell(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
        
                    _, preds = torch.max(outputs, 1)
                    val_accuracy += sum((preds == labels).cpu().numpy())
                    val_batch_num += 1

                val_loss /= len(test_dl)
                val_accuracy = val_accuracy/len(test_dataset)
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

                modell.train()
        
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
            batch_num += 1
    
    torch.cuda.empty_cache()
    print('Training complete..')
    return modell, history

def save_history(modell, history, split, data_ratio, mode):
    """
        A simple function that saves the histories and the clasificator
    """
    FILEBASE = f"Histories/{modell.get_name}_model_{split}_split_{data_ratio}_ratio_{mode}_mode"
    torch.save(modell.state_dict(), FILEBASE + '.pt')
    with open(FILEBASE + '-history.pkl', 'wb') as file:
        pickle.dump(history, file)
        print(f'{FILEBASE} instance saved')

def Test_with_all_ratios(split, epochs, network_name, mode, geoaugment):
    """
        This simple function manages the training with the given parameters with all data_ratios 
        (This is just a helper function to make training simpler and more manageble)
        split (str): choose one from ['orig','0','1', '2','3']
        epchs (int): number of epochs
        network_name (str): choose one from ['resnet', 'vgg', efficient]
                            'resnet'    --> Resnet18
                            'vgg'       --> Vgg16  
                            'efficeint' --> EfficientNet_b0
        mode (str): means of dataset augmentation can be one from ['oversapmling', 'gan'] or None:
                            'oversampling' --> simple oversampling (doubleing the images for the incomplete class)
                            'gan'          --> using pretrained GANs for generating missing images
                            None           --> No augmentation, this is for benchmark
    """
    
    BATCH_SIZE = 128
    if mode is None:
      #Train on the whole dataset only, if there is no oversampling or gan augmentation
      start = time.time()
      
      network = get_model(network_name).to(device)
      print(f"Working on {network.get_name}_model_{split}_split_{1}_ratio_{mode}_mode")
      print("Models are on cuda: ",next(network.parameters()).is_cuda )
      
      optimizer = torch.optim.Adam(network.parameters(), lr=3e-5)
      train_dataset, test_dataset = DatasetMaker(split = split, mode = mode, data_ratio = 1, geoaugment = geoaugment) 
      network, history = train(epochs = epochs, modell=network, loss_fn=LOSS_FN, optimizer = optimizer, train_dataset = train_dataset, test_dataset = test_dataset, batch_size = BATCH_SIZE,  shuffle = True)
      save_history(network, history, split, 1, mode)
      
      end = time.time()
      print("Time it took to complete: ", end - start)

    #0.8
    start = time.time()
    network = get_model(network_name).to(device)
    print(f"Working on {network.get_name}_model_{split}_split_{0.8}_ratio_{mode}_mode")
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-5)
    train_dataset, test_dataset = DatasetMaker(split = split, mode = mode, data_ratio = 0.8, geoaugment = geoaugment) 
    network, history = train(epochs = epochs, modell=network, loss_fn=LOSS_FN, optimizer = optimizer, train_dataset = train_dataset, test_dataset = test_dataset, batch_size = BATCH_SIZE,  shuffle = True)
    save_history(network, history, split, 0.8, mode)
    
    end = time.time()
    print("Time it took to complete: ", end - start)

    #0.6
    start = time.time()
    network = get_model(network_name).to(device)
    print(f"Working on {network.get_name}_model_{split}_split_{0.6}_ratio_{mode}_mode")
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-5)
    train_dataset, test_dataset = DatasetMaker(split = split, mode = mode, data_ratio = 0.6, geoaugment = geoaugment) 
    network, history = train(epochs = epochs, modell=network, loss_fn=LOSS_FN, optimizer = optimizer, train_dataset = train_dataset, test_dataset = test_dataset, batch_size = BATCH_SIZE,  shuffle = True)
    save_history(network, history, split, 0.6, mode)
    
    end = time.time()
    print("Time it took to complete: ", end - start)

    #0.4
    start = time.time()
    network = get_model(network_name).to(device)
    print(f"Working on {network.get_name}_model_{split}_split_{0.4}_ratio_{mode}_mode")
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-5)
    train_dataset, test_dataset = DatasetMaker(split = split, mode = mode, data_ratio = 0.4, geoaugment = geoaugment) 
    network, history = train(epochs = epochs, modell=network, loss_fn=LOSS_FN, optimizer = optimizer, train_dataset = train_dataset, test_dataset = test_dataset, batch_size = BATCH_SIZE,  shuffle = True)
    save_history(network, history, split, 0.4, mode)
    
    end = time.time()
    print("Time it took to complete: ", end - start)

    #0.2
    start = time.time()
    print(f"Working on {network.get_name}_model_{split}_split_{0.2}_ratio_{mode}_mode")
    network = get_model(network_name).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-5)
    train_dataset, test_dataset = DatasetMaker(split = split, mode = mode, data_ratio = 0.2, geoaugment = geoaugment) 
    network, history = train(epochs = epochs, modell=network, loss_fn=LOSS_FN, optimizer = optimizer, train_dataset = train_dataset, test_dataset = test_dataset, batch_size = BATCH_SIZE,  shuffle = True)
    save_history(network, history, split, 0.2, mode)
    
    end = time.time()
    print("Time it took to complete: ", end - start)
    

if __name__=='__main__':
    splits = ['orig','0','1', '2','3']

    for split in splits:
        #simple tests when no augmentation is done
        Test_with_all_ratios(split = split, epochs = 1, network_name='resnet', mode = None, geoaugment=False)
        Test_with_all_ratios(split = split, epochs = 1, network_name='vgg', mode = None, geoaugment=False)
        Test_with_all_ratios(split = split, epochs = 1, network_name='efficient', mode = None, geoaugment=False)

        #simple tests when oversasmpling is used to balance classes
        Test_with_all_ratios(split = split, epochs = 1, network_name='resnet', mode = 'oversampling', geoaugment=False)
        Test_with_all_ratios(split = split, epochs = 1, network_name='vgg', mode = 'oversampling', geoaugment=False)
        Test_with_all_ratios(split = split, epochs = 1, network_name='efficient', mode = 'oversampling', geoaugment=False)

        #simple tests when gan is used to balance classes
        Test_with_all_ratios(split = split, epochs = 1,  network_name='resnet', mode = 'gan', geoaugment=False)
        Test_with_all_ratios(split = split, epochs = 1, network_name='vgg', mode = 'gan', geoaugment=False)
        Test_with_all_ratios(split = split, epochs = 1, network_name='efficient', mode = 'gan', geoaugment=False)

        #simple Test when just geoaugmentation is used
        #Test_with_all_ratios(split = split, epochs = 1, network_name='resnet', mode = None, geoaugment=True)
        #Test_with_all_ratios(split = split, epochs = 1, network_name='vgg', mode = None, geoaugment=True)
        #Test_with_all_ratios(split = split, epochs = 1, network_name='efficient', mode = None, geoaugment=True)

        #simple tests when oversasmpling is used to balance classes and geoaugment
        #Test_with_all_ratios(split = split, epochs = 1, network_name='resnet', mode = 'oversampling', geoaugment=True)
        #Test_with_all_ratios(split = split, epochs = 1, network_name='vgg', mode = 'oversampling', geoaugment=True)
        #Test_with_all_ratios(split = split, epochs = 1, network_name='efficient', mode = 'oversampling', geoaugment=True)

        #simple tests when gan is used to balance classes and geoaugment
        #Test_with_all_ratios(split = split, epochs = 1, network_name='resnet', mode = 'gan', geoaugment=True)
        #Test_with_all_ratios(split = split, epochs = 1, network_name='vgg', mode = 'gan', geoaugment=True)
        #Test_with_all_ratios(split = split, epochs = 1, network_name='efficient', mode = 'gan', geoaugment=True)
    
    print("All training is done")
    