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

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

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

batch_size = 10

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

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

# Save checkpoint
if os.path.exists(pth_path):
    epoch = auto_load_resume(model, pth_path, status='test')
else:
    sys.exit('There is not a pth exist.')

target_layers = [model.shortcut]
counter = 0

for data in testloader:
    if counter < 579:
        batch_size = 10
    else:
        batch_size = 4
    for i in range(batch_size):

        input_tensor, _, _, _, rgb_input_path = data
        input_tensor = input_tensor.to(DEVICE)


        rgb_input = cv2.imread(rgb_input_path[i], 1)[:, :, ::-1]

        if len(rgb_input.shape) == 2:
            rgb_input = np.stack([rgb_input] * 3, 2)
        rgb_input = Image.fromarray(rgb_input, mode='RGB')
        rgb_input = transforms.Resize((448, 448), Image.BILINEAR)(rgb_input)
        rgb_input = np.float32(rgb_input) / 255
        #print("NAME")
        #print(rgb_input_path[0].split('/')[5])
        targets = None
        cam_algorithm = methods["gradcam"]
        with cam_algorithm(model=model,
                            target_layers=target_layers,
                            use_cuda=True) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            input_tensor = input_tensor[i:i+1]
            #input_tensor = input_tensor.to(DEVICE)
    
            grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)
    
            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
                

            #result = grayscale_cam > 0.7
            #result2 = grayscale_cam > 0.8
            #result3 = grayscale_cam > 0.6
            #print("Shape is",grayscale_cam.shape)
            #print("PRINT DIM",result[0].shape, result[1].shape)

            cam_image = show_cam_on_image(rgb_input, grayscale_cam, use_rgb=True)
            #cam2_image = show_cam_on_image(rgb_input, result, use_rgb=True)
            #cam3_image = show_cam_on_image(rgb_input, result2, use_rgb=True)
            #cam4_image = show_cam_on_image(rgb_input, result3, use_rgb=True)
    
            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            #cam2_image = cv2.cvtColor(cam2_image, cv2.COLOR_RGB2BGR)
            #cam3_image = cv2.cvtColor(cam3_image, cv2.COLOR_RGB2BGR)
            #cam4_image = cv2.cvtColor(cam4_image, cv2.COLOR_RGB2BGR)
          

            ##result = np.where(grayscale_cam > 0.7)

        cv2.imwrite(os.path.join('/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/heatmaps', rgb_input_path[i].split('/')[5]), cam_image)
        #cv2.imwrite(rgb_input_path[i].split('/')[5], cam_image)
        #cv2.imwrite(f'{"gradcam"}{counter}{"-"}{i}_cam2.jpg', cam2_image)
        #cv2.imwrite(f'{"gradcam"}{counter}{"-"}{i}_cam3.jpg', cam3_image)
        #cv2.imwrite(f'{"gradcam"}{counter}{"-"}{i}_cam4.jpg', cam4_image)
        #cv2.imwrite(f'{"gradcam"}{counter}{"-"}{i}_gb.jpg', gb)
        #cv2.imwrite(f'{"gradcam"}{counter}{"-"}{i}_cam_gb.jpg', cam_gb)    

    counter = counter + 1



