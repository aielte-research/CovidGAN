import random
import sys
import os
import pickle
from sklearn.model_selection import train_test_split

classes = ['COVID-19', 'Non-COVID', 'Normal']
class_to_idx = {cls_name : i for i, cls_name in enumerate(classes)}

train_subset_file = "Test_orig/orig_split_train_and_val.pkl"
test_subset_file = "Test_orig/orig_split_test.pkl"
split = "7_nested"

def save_sets(train_set, val_set, test_set, split):
  dump_obj1 = (train_set, val_set)
  dump_obj2 = test_set
  
  with open(f'{split}_split_train_and_val.pkl', 'wb') as file:
      pickle.dump(dump_obj1, file)
    
  with open(f'{split}_split_test.pkl', 'wb') as file:
    pickle.dump(dump_obj2, file)



test_dataset = []
path = os.path.join('./output/data/COVID_QU/Indice_files', test_subset_file)
with open(path, 'rb') as file:
  data = pickle.load(file)
  test_dataset = [ (x[0],x[1]) for x in data]
  
train_dataset = []
path = os.path.join('./output/data/COVID_QU/Indice_files', train_subset_file)
with open(path, 'rb') as file:
  data = pickle.load(file)
  data = [*data[0],*data[1]] 
  train_dataset = [ (x[0],x[1]) for x in data]
  
dataset = [*train_dataset, *test_dataset]


trainey, test = train_test_split(dataset, test_size=0.2, random_state=42+7)
val_size = 0.16*(5/4) #should be 16% of all data
train, val = train_test_split(trainey, test_size=val_size, random_state=420+2*7)
save_sets(train, val, test, split)

covid = [x for x in trainey if x[1] == class_to_idx['COVID-19']]
ratios = {0.8 : 4/5, 
          0.6 : 3/4, 
          0.4: 2/3,
          0.2: 1/2}
with open(f'{split}_split_{1}_gan.pkl', 'wb') as file:
    pickle.dump(covid, file)
    
gan = covid
for ratio, value in ratios.items():
    _, gan = train_test_split(gan, test_size=value, random_state=420+3*4)
    print(f"Saving {len(gan)} instance for {ratio}")
    with open(f'{split}_split_{ratio}_gan.pkl', 'wb') as file:
        pickle.dump(gan, file)
print("Done with saving gans")
