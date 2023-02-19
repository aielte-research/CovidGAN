#This file should generate a subset list of all the data with a given sampling ratio 
#for reproducibility issue 
# (This will generate a text file which you can later use, to load the exact same dataset from your earlier training)
from data.covid_qe_data_loader import CovidQuPositiveDataSet
from random import Sample
from math import round
import sys

ratio = sys.argv[1]
root = './output/data'
dataset = CovidQuPositiveDataSet(root)
num_of_samples = len(round(dataset*ratio))
indicies = Sample(range(len(dataset)), num_of_samples)
imgs = [dataset.get_name(i) for i in indicies]
filename=f'indices_{ratio}.txt'
with open(filename, 'w') as file:
    for i in imgs:
        file.write(i)
        file.write('\n')
print("Done with making {}", filename)
