import torch
import os
import torchvision
import numpy as np
from torch.utils.data import Dataset
from model_birding_net import BirdingNet
from model_create_dataset import FeatureMapsDataset
from torch import tensor
from pytorch_grad_cam import GradCAM
from grad_cam_new_2 import GradCAM_New_2
import pickle
import math
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = BirdingNet()
model.load_state_dict(torch.load('/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/model_best_test_f1.pth'))
model.to(device)

testset = FeatureMapsDataset(path = '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/npy_saved_files_test/')
batch_size = 1
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=8, drop_last=False)

target_layers = [model.shortcut]

with open('s_values_dict.pkl', 'rb') as fp:
    s_values_dict = pickle.load(fp)

with open('vector_MMAL_dict.pkl', 'rb') as fp:
    vector_MMAL_dict = pickle.load(fp)

attribute_names = {}
with open("/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/datasets/CUB_200_2011/attributes/attributes.txt") as f:
    for line in f:
       (key, val) = line.split()
       attribute_names[int(key)] = val
print("OK")
with open("/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/results_con5b_lay/results.txt", mode="wt") as f:
    for data in testloader:
        input_tensor, attribute_vector, data_path = data
        input_tensor = input_tensor.to(device)
        attribute_vector = attribute_vector.to(device)
    
    
        input_tensor.requires_grad = True   
        distance_dict = {}
        targets = None
        with GradCAM_New_2(model=model,
                            target_layers=target_layers,
                            use_cuda=True) as cam:
       
            for i in range (312):       
                grayscale_cam_att = cam(input_tensor=input_tensor,
                                    targets=targets,
                                    target_categories=[i],
                                    aug_smooth=False,
                                    eigen_smooth=False)
                result_grayscale = np.squeeze(grayscale_cam_att[0])
                total_vector = s_values_dict[data_path[0]]*result_grayscale

                # Dictionary with distances 
                distance_dict[math.dist(vector_MMAL_dict[data_path[0]], total_vector)] = i+1  # i+1 because attributes list starts from 1

        # Sort keys of the dictionary       
        myKeys = list(distance_dict.keys())  
        myKeys.sort()

    
        #f.write("This is text!\n")
        #f.write("And some more text\n")
    # Print the 5 most important attributes of each img
        #print('\n')
        f.write(data_path[0])
       
        #print("Five most important attributes are:")
        for j in range (5):
            f.write('#')
            f.write(attribute_names[distance_dict[myKeys[j]]])
        f.write('\n')
f.close()
    #sorted_dict = {j: distance_dict[j] for j in myKeys}
    #print(sorted_dict)









             
             

   
 
             







