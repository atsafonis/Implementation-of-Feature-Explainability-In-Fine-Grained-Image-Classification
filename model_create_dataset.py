import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
import glob

class FeatureMapsDataset (Dataset) : 
    def __init__(self, path):
        self.path = path
        self.file_list = glob.glob(self.path + "*")
        self.attributes_path = '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/datasets/CUB_200_2011/attributes/image_attribute_labels.txt'
        self.images_path = '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/datasets/CUB_200_2011/images.txt'
        self.attributes_labels = np.loadtxt(self.attributes_path,dtype="int",usecols =(0,1,2))
        self.images = np.loadtxt(self.images_path, delimiter=' ', dtype=str)
        self.img_value_dict = {}
        self.attributes_dict = {}

        for i in range (len(self.images)):
            self.img_value_dict[self.images[i][1].split('/')[1]] = int(self.images[i][0])

        for i in range (len(self.attributes_labels)):
            if self.attributes_labels[i][0] in self.attributes_dict:
                if self.attributes_labels[i][2] == 1:
                    self.attributes_dict[self.attributes_labels[i][0]][self.attributes_labels[i][1] - 1] = 1
            else:
                self.attributes_dict[self.attributes_labels[i][0]] = np.zeros(312, dtype='f')
                if self.attributes_labels[i][2] == 1:
                    self.attributes_dict[self.attributes_labels[i][0]][self.attributes_labels[i][1] - 1] = 1  
            
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data_path = self.file_list[idx]
        data = np.load(data_path, allow_pickle = True)
        feature_maps = data[1]
        feature_maps_tensor = torch.from_numpy(feature_maps)
      
        id = self.img_value_dict[data_path.split('/')[7][:-4]]
        vector = self.attributes_dict[id]
        vector_tensor = torch.from_numpy(vector)
        
        return feature_maps_tensor, vector_tensor, data_path.split('/')[7][:-4]