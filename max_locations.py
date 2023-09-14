import torch
import torch.nn as nn
from numpy import save
import sys
from tqdm import tqdm
from config import input_size, root, proposalN, channels
from utils.read_dataset import read_dataset
from utils.auto_laod_resume import auto_load_resume
from networks.model import MainNet
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import cv2
import numpy as np
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset
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

# Load dataset
_, testloader = read_dataset(input_size, batch_size, root, set)

methods = \
    {"gradcam": GradCAM,
     "hirescam": HiResCAM,
     "scorecam": ScoreCAM,
     "gradcam++": GradCAMPlusPlus,
     "ablationcam": AblationCAM,
     "xgradcam": XGradCAM,
     "eigencam": EigenCAM,
     "eigengradcam": EigenGradCAM,
     "layercam": LayerCAM,
     "fullgrad": FullGrad,
     "gradcamelementwise": GradCAMElementWise}

# Model
model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)
model = model.to(device)
#criterion = nn.CrossEntropyLoss()

# Save checkpoint
if os.path.exists(pth_path):
    epoch = auto_load_resume(model, pth_path, status='test')
else:
    sys.exit('There is not a pth exist.')

target_layers = [model.shortcut]
dict_max_locations = {}

for data in testloader:
    
    input_tensor, _, _, _, rgb_input_path = data
    input_tensor = input_tensor.to(device)

    targets = None
    cam_algorithm = methods["gradcam"]
    with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=True) as cam:
        #cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)
    
    grayscale_cam = np.squeeze(grayscale_cam, axis=None)
    max_location = np.unravel_index(grayscale_cam.argmax(), grayscale_cam.shape)
    dict_max_locations[rgb_input_path[0].split('/')[5]] = max_location
    


with open('max_locations.pkl', 'wb') as fp:
    pickle.dump(dict_max_locations, fp)
