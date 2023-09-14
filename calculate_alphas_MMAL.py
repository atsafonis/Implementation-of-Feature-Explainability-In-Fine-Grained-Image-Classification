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
from pytorch_grad_cam import GradCAM
from grad_cam_new import GradCAM_New
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

set = 'CUB'
if set == 'CUB':
    root = './datasets/CUB_200_2011'  # dataset path
    # Model path
    pth_path = "./models/cub_epoch144.pth"
    num_classes = 200
elif set == 'Aircraft':
    root = './datasets/FGVC-aircraft'  # dataset path
    # Model path
    pth_path = "./models/air_epoch146.pth"
    num_classes = 100

batch_size = 1

_, testloader = read_dataset(input_size, batch_size, root, set)

model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)
model = model.to(device)
#criterion = nn.CrossEntropyLoss()


if os.path.exists(pth_path):
    epoch = auto_load_resume(model, pth_path, status='test')
else:
    sys.exit('There is not a pth exist.')


target_layers = [model.shortcut]

dict_MMAL_values = {}
s_values_dict = {}
vector_MMAL_dict = {}

with open('max_locations.pkl', 'rb') as fp:
    max_locations = pickle.load(fp)

for data in testloader:
    
    input_tensor, _, _, _, rgb_input_path = data
    input_tensor = input_tensor.to(device)


    targets = None
    with GradCAM_New(model=model,
                        target_layers=target_layers,
                        use_cuda=True) as cam:

        
        #cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)
    
    #Save in a dictionary the a values from changed gradcam
    dict_MMAL_values[rgb_input_path[0].split('/')[5]] = grayscale_cam

    #Return conv5b from MMAL model forward 
    feature_map = model.feature_for(input_tensor)

    #Scaling to fit max locations from oroginal gradcam with (14,14) img dimensions after MMAL model
    scale_height = input_tensor.shape[2]/feature_map.shape[2]         #input_tensor.shape[2] = 448, feature_map.shape[2] = 14
    height = round((max_locations[rgb_input_path[0].split('/')[5]][0] + 1)/scale_height) - 1
    scale_width = input_tensor.shape[3]/feature_map.shape[3]
    width = round((max_locations[rgb_input_path[0].split('/')[5]][1] + 1)/scale_width) - 1

    feature_map = np.squeeze(feature_map)
    
    #Put max locations from original gradcam into feature maps to get the s values 
   
    s_values_dict[rgb_input_path[0].split('/')[5]] = feature_map[:, height, width].cpu().detach().numpy()
   
    result_grayscale = np.squeeze(grayscale_cam[0])

    #This is the vector with a values multiplied by s values for the 2048 feature maps with shape (2048,)
    total_vector = s_values_dict[rgb_input_path[0].split('/')[5]] * result_grayscale
    
    #Dictionary with a*s values 
    vector_MMAL_dict[rgb_input_path[0].split('/')[5]] = total_vector 
    #print(vector_MMAL_dict)

with open('s_values_dict.pkl', 'wb') as fp:
    pickle.dump(s_values_dict, fp)

with open('vector_MMAL_dict.pkl', 'wb') as fp:
    pickle.dump(vector_MMAL_dict, fp)
           
print("END")


