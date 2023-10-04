import torch
import os
import torchvision
import numpy as np
from torch.utils.data import Dataset
from model_att_net import ATT_Net
from model_create_dataset import FeatureMapsDataset
from torch import tensor
from pytorch_grad_cam import GradCAM
from grad_cam_new_2 import GradCAM_New_2
import pickle
import math
import sys
import timeit
from numpy.linalg import norm


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ATT_Net()
model.load_state_dict(torch.load('/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/model_best_test_f1_layer4.pth'))
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
with open("/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/results_con5b_lay/best_results_312.txt", mode="wt") as f:
    for data in testloader:
        

        input_tensor, attribute_vector, data_path = data
        input_tensor = input_tensor.to(device)
        attribute_vector = attribute_vector.to(device)


        input_tensor.requires_grad = True   
        distance_dict = {}
        targets = None
        avg_time = 0
        with GradCAM_New_2(model=model,
                            target_layers=target_layers,
                            use_cuda=True) as cam:
       
            #for i in range (312):      
                #Start the time here
                #start = timeit.default_timer() 
                grayscale_cam_att = cam(input_tensor=input_tensor.repeat((312,1,1,1)),
                                    targets=targets,
                                    #target_categories= list(range(0, 312)),  ##[i]
                                    aug_smooth=False,
                                    eigen_smooth=False)


                ##stop and print time
                #stop = timeit.default_timer()
                #time = stop - start
                #print(time/312)
                
                ##result_grayscale = np.squeeze(grayscale_cam_att[0])
                #total_vector = s_values_dict[data_path[0]]*grayscale_cam_att[0]

        total_att_result = []
        ## grayscale_cam_att[0] is the list of importances for every attribute for 2048 feature maps
        
        for element in grayscale_cam_att[0]:
            total_att_result.append(element * s_values_dict[data_path[0]])

        for i in range (312):
            # Dictionary with distances 
            #distance_dict[math.dist(vector_MMAL_dict[data_path[0]], total_att_result[i])] = i+1  # i+1 because attributes list starts from 1
            distance_dict[np.dot(vector_MMAL_dict[data_path[0]], total_att_result[i])/norm(vector_MMAL_dict[data_path[0]])*norm(total_att_result[i])] = i+1
            #cosine = np.dot(A,B)/(norm(A)*norm(B))
        # Sort keys of the dictionary       
        myKeys = list(distance_dict.keys())  
        myKeys.sort()

    # Print the 5 most important attributes of each img
    
        f.write(data_path[0])
       
        #print("Five most important attributes are:")
        for j in range (312):
            f.write('#')
            f.write(attribute_names[distance_dict[myKeys[j]]])
        f.write('\n')
f.close()
    





             
             

   
 
             







