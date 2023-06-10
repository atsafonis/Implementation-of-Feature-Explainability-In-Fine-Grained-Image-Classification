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


target_layers = [model.feature_for]

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
    print(grayscale_cam)
