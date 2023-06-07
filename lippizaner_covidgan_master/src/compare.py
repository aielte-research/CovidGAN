import os
import pickle

test_subset_file = "Test_6_nested/6_nested_split_test.pkl"

test_dataset = []
path = os.path.join('./output/data/COVID_QU/Indice_files', test_subset_file)
with open(path, 'rb') as file:
  data = pickle.load(file)
  test_dataset = [ (x[0],x[1]) for x in data]
  
test_set_1 = set(test_dataset)

test_subset_file = "Test_7_nested/7_nested_split_test.pkl"

test_dataset = []
path = os.path.join('./output/data/COVID_QU/Indice_files', test_subset_file)
with open(path, 'rb') as file:
  data = pickle.load(file)
  test_dataset = [ (x[0],x[1]) for x in data]
  
test_set_2 = set(test_dataset)

print(test_set_1 == test_set_2)