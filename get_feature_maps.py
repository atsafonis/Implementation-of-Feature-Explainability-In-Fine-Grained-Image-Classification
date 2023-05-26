import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from config import input_size, root, proposalN, channels
from utils.read_dataset import read_dataset
from utils.auto_laod_resume import auto_load_resume
from networks.model import MainNet
import numpy as np
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# dataset
set = 'CUB'
if set == 'CUB':
    root = './datasets/CUB_200_2011'  # dataset path
    # model path
    pth_path = "./models/cub_epoch144.pth"
    num_classes = 200
elif set == 'Aircraft':
    root = './datasets/FGVC-aircraft'  # dataset path
    # model path
    pth_path = "./models/air_epoch146.pth"
    num_classes = 100

batch_size = 10

# load dataset
trainloader, testloader = read_dataset(input_size, batch_size, root, set)

# model
model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

# checkpoint
if os.path.exists(pth_path):
    epoch = auto_load_resume(model, pth_path, status='test')
else:
    sys.exit('There is not a pth exist.')


for data in trainloader:

    input_tensor, labels, _, _, rgb_input_path = data
    input_tensor = input_tensor.to(DEVICE)

    feature_map = model.feature_map_for(input_tensor)
         
    np_features = feature_map.cpu().detach().numpy() 
    labels = labels.cpu().detach().numpy()
        
    for i in range(input_tensor.shape[0]):
        
        input_list = [labels[i], np_features[i]]
       
        np.save('/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/npy_saved_files_train/' + str(os.path.basename(os.path.normpath(rgb_input_path[i]))), input_list)

for data in testloader:

    input_tensor, labels, _, _, rgb_input_path = data
    input_tensor = input_tensor.to(DEVICE)

    feature_map = model.feature_map_for(input_tensor)
         
    np_features = feature_map.cpu().detach().numpy() 
    labels = labels.cpu().detach().numpy()
        
    for i in range(input_tensor.shape[0]):
        
        input_list = [labels[i], np_features[i]]
       
        np.save('/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/npy_saved_files_test/' + str(os.path.basename(os.path.normpath(rgb_input_path[i]))), input_list)        
    


     
