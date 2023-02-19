import logging
import pathlib
import os 

from helpers.configuration_container import ConfigurationContainer
from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor, Compose, Resize, Grayscale, RandomHorizontalFlip, RandomAffine, Normalize
from torch.utils.data import Dataset
from data.data_loader import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image

from PIL import Image
import torch
from torch.autograd import Variable
import random
import math
import pickle


WIDTH = 128
HEIGHT = 128
CHANNELS = 3 

class CoviudQuPositiveDataLoader(DataLoader):
    def __init__(self, use_batch=True, batch_size=1, n_batches=0, shuffle=False):
        super().__init__(CovidQuPositiveDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return WIDTH*HEIGHT #*CHANNELS

    @staticmethod
    def save_images(images, shape, filename):
         # img_view = data.view(num_images, 1, WIDTH, HEIGHT)
        img_view = images.view(images.size(0), 1, WIDTH, HEIGHT)
        # img_view = images.view(images)
        save_image(make_grid(img_view), filename)

class CovidQuPositiveDataSet(ImageFolder):
    """
      This is a Grayyscale 128*128 dataset, with only COVID_19 pictures
      Dataset is COVID_Qu_Ex dataset from Kaggle
    """
    
    _logger = logging.getLogger(__name__)
    base_folder = 'COVID_QU/all_covid'
    
    """def __init__(self, root, transform=None, target_transform=None, **kwargs):
        target_dir=os.path.join(root, self.base_folder)
        transforms = Compose([Grayscale(num_output_channels=1), Resize(size=[HEIGHT, WIDTH], interpolation=Image.NEAREST), ToTensor()])
        try:
            super().__init__(target_dir, transforms, target_transform)
        except Exception as ex:
            self._logger.critical("An error occured while trying to load CovidQu: {}".format(ex))
            raise ex"""
            
    def __init__(self, root, transform=None, target_transform=None, **kwargs):
        def get_images(class_name):
            images = [(os.path.join(self.image_dirs[class_name], x),0) for x in os.listdir(self.image_dirs[class_name]) if x.lower().endswith('png')]
            #print(f'Found {len(images)} {class_name} examples')
            return images
                              
        self.cc = ConfigurationContainer.instance()
        settings = self.cc.settings['dataloader']
        subset_file = settings.get('subset_file',None)
        
        target_dir=os.path.join(root, self.base_folder)
        self.image_dirs = {}
        self.image_dirs['covid'] = os.path.join(target_dir, 'COVID_19')
        
        images = []
        self.class_names = ['covid']
        
        #######
        #TODO rewrite this
        #######
        
        if subset_file is not None:
          path = os.path.join(root, 'COVID_QU/Indice_files', subset_file)
          with open(path, 'rb') as file:
            data = pickle.load(file)
            images = [ (os.path.join(self.image_dirs['covid'], x[0]),x[1]) for x in data]
            print("Succesfully extracted covid images: ", len(images))
        else:
              images = get_images('covid') #If there was no subset file, use all images
            
        
        #transform = [Resize(size=[HEIGHT, WIDTH], interpolation=Image.NEAREST), 
        #           ToTensor(),
        #            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #            Grayscale(num_output_channels=1)]
        transform = [Grayscale(num_output_channels=1), Resize(size=[HEIGHT, WIDTH], interpolation=Image.NEAREST), ToTensor()]
        if settings.get('augment', False) == True:
            geo = [RandomAffine(3) ]# RandomHorizontalFlip()]
            for i in geo:
              transform.append(i)
        transforms = Compose(transform)
        target_dir = os.path.join(root, self.base_folder)
        super().__init__(target_dir, transforms, target_transform)
        self.samples = images
        self.imgs = images
        
    """def __len__(self):
        return self.length
    
    
    def __getitem__(self, index):
        #print("Called getitem")
        class_name = 'covid' #random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        with open(image_path, "rb") as f:
          image = Image.open(f).convert("RGB")
        #image = read_image(image_path)
        #image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)"""
        
    def get_name(self, index):
        index = index % len(self.images['covid'])
        image_name = self.images['covid'][index]
        return image_name
          


    


