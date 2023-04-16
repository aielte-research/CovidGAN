import os
import subprocess
import pickle

import torch

torch.set_num_threads(8)

gan_directories = {
    'Test_orig_0.8' : {'dir':'2023-03-14_19-00-08','best':'127.0.0.1-5002'},
    'Test_orig_0.6' : {'dir':'2023-03-15_09-51-22','best':'127.0.0.1-5001'},
    'Test_orig_0.4' : {'dir':'2023-03-16_17-23-13','best':'127.0.0.1-5002'},
    'Test_orig_0.2' : {'dir':'2023-03-17_11-28-34','best':'127.0.0.1-5002'},
    
    'Test_0_0.8' : {'dir':'2023-03-23_15-19-22','best':'127.0.0.1-5002'},
    'Test_0_0.6' : {'dir':'2023-03-24_16-23-49','best':'127.0.0.1-5001'},
    'Test_0_0.4' : {'dir':'2023-03-25_09-50-59','best':'127.0.0.1-5001'},
    'Test_0_0.2' : {'dir':'2023-03-25_16-58-45','best':'127.0.0.1-5003'},

    'Test_1_0.8' : {'dir':'2023-03-21_16-13-32','best':'127.0.0.1-5003'},
    'Test_1_0.6' : {'dir':'2023-03-22_09-56-24','best':'127.0.0.1-5001'},
    'Test_1_0.4' : {'dir':'2023-03-22_18-38-27','best':'127.0.0.1-5000'},
    'Test_1_0.2' : {'dir':'2023-03-23_08-16-51','best':'127.0.0.1-5002'},

    'Test_2_0.8' : {'dir':'2023-03-19_18-20-08','best':'127.0.0.1-5001'},
    'Test_2_0.6' : {'dir':'2023-03-20_08-46-21','best':'127.0.0.1-5001'},
    'Test_2_0.4' : {'dir':'2023-03-20_16-58-27','best':'127.0.0.1-5001'},
    'Test_2_0.2' : {'dir':'2023-03-21_08-49-59','best':'127.0.0.1-5002'},

    'Test_3_0.8' : {'dir':'2023-03-17_17-42-29','best':'127.0.0.1-5001'},
    'Test_3_0.6' : {'dir':'2023-03-18_07-56-44','best':'127.0.0.1-5003'},
    'Test_3_0.4' : {'dir':'2023-03-18_23-01-23','best':'127.0.0.1-5002'},
    'Test_3_0.2' : {'dir':'2023-03-19_07-43-42','best':'127.0.0.1-5000'},
     }
idx_to_class ={
        0: 'covid',
        1: 'viral',
        2: 'normal'
    }

def get_needed_size(split, ratio):
    all_img_file = f'Lung_Segmentation_Data/Indicies_files/Test_{split}/{split}_split_train_and_val.pkl'
    imgs = load_images_from_file(all_img_file)
    imgs = [*imgs[0],*imgs[1]]
    non_cov_imgs = []
    for x in imgs:
        if idx_to_class[x[1]] != "covid": non_cov_imgs.append(x)
        
    #print("Number of non_cov imgs: ",len(non_cov_imgs))
    
    case_specific_cov_file = f'Lung_Segmentation_Data/Indicies_files/Test_{split}/{split}_split_{ratio}_gan.pkl'
    cov_imgs = load_images_from_file(case_specific_cov_file)
    missing = (len(non_cov_imgs))//2 - len(cov_imgs) +1
    return missing

def load_images_from_file(file):
    with open(file, 'rb') as file:
        data = pickle.load(file)
    return data   

def generate_images_to_dir(split, data_ratio, directory, size):
    """
        Generates pictures with a given gan, to a given directory
    """
    #Goes into Lipizzaner's directory and then generates images with a given GAN (this function is a Lipizzaner built-in method)
    #Then returns into this directory
    
    curr_dir = os.getcwd()
    
    lippi_dir = '/home/bbernard/CovidGan-elte/CovidGAN/lippizaner_covidgan_master/src/'  #Change
    
    output_dir = os.path.join(curr_dir, directory) #?
   
   #Gan to use is determined by the split and the data_ratio parameters
    gan_dir = gan_directories[f'Test_{split}_{data_ratio}']
    src_dir = os.path.join(lippi_dir, f'output/lipizzaner_gan/master/{gan_dir["dir"]}/{gan_dir["best"]}')
    
    config_file = os.path.join(lippi_dir, f'configuration/covid-qu-conv/Test_{split}/covidqu_{data_ratio}.yml')
    
    #man = os.path.join(lippi_dir, 'main.py')

    code =f'python main.py generate --mixture-source {src_dir} -o {output_dir} --sample-size {size} -f {config_file}'
    os.chdir(lippi_dir)
    subprocess.run(code, shell=True)
    os.chdir(curr_dir)

if __name__=='__main__':
    splits = ["orig", '0', '1', '2', '3']
    ratios = [0.8, 0.6, 0.4, 0.2]
    for split in splits:
        for ratio in ratios:
            output_dir = f'Lung_Segmentation_Data/generated/Test_{split}/gan_{ratio}'
            size = get_needed_size(split, ratio)
            generate_images_to_dir(split, ratio, output_dir, size)
            print(f"Generated {size} for {split}_{ratio}")
