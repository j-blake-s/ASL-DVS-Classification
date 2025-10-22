import os
import numpy as np
from torch.utils.data import Dataset


# Grab list of files from directory
def get_files(path, person, dataset): 
  assert person=='blake' or person=='james'or person=='peyton', "Person must be one of [blake / james / peyton]"
  assert dataset=='dvs' or dataset=='rgb', "Dataset must be one of [ rgb / dvs ]"
  path = os.path.join(path, dataset, person)
  return [os.path.join(path, f) for f in os.listdir(path)]

# Grab files from all users
def get_all(path, dataset):
  blake = get_files(path, 'blake', dataset)
  james = get_files(path, 'james', dataset)
  peyton = get_files(path, 'peyton', dataset)

  return (blake, james, peyton)

# Wrapper class for ASL dataset
class ASL(Dataset):
  def __init__(self, files, augment=None, print_func=None, combine_classes=False):
    
    self.size = len(files)
    self.videos = None
    self.labels = np.zeros(shape=(self.size,))
    self.aug = augment

    for i, fn in enumerate(files):
      with np.load(fn) as f:
        if self.videos is None: self.videos = np.zeros(shape=(self.size,) + f['x'].shape) 
        self.videos[i] = f['x']
        self.labels[i] = f['y'][0]
      if print_func is not None: print_func((i+1)/self.size)

  def __len__(self): return self.size
  def __getitem__(self, idx): 
    vid = self.videos[idx]
    vid = np.transpose(vid, (2, 0, 1, 3))
    if self.aug is not None: vid = self.aug(vid)
    return vid, self.labels[idx]


# Load data from files
def load_data(train, test, augment=None, verbose=False, combine_classes=False):
  print("Loading Dataset...")
  def progress_meter(pre): return lambda x: print(f"{pre}({int(100*x)}%)",end=("\r" if x < 1 else "\n"))
  if train is not None:
    train_data = ASL(train, augment=augment, combine_classes=combine_classes, print_func=(progress_meter("\tLoading Train Dataset...") if verbose else None))
  else:
    train_data = None
  
  if test is not None:
    test_data = ASL(test, combine_classes=combine_classes, print_func=(progress_meter("\tLoading Test Dataset...")if verbose else None))
  else:
    test_data = None
  return train_data, test_data

