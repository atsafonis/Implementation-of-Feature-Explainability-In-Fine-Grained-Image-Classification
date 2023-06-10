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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

dict = {}
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

        ##feature_map = model.feature_for(input_tensor)
        print(input_tensor.shape)
        feature_map = model.feature_map_for(input_tensor)
        #print(feature_map.dtype)
        print(feature_map.shape)
        ###print(feature_map)

        #print(len(result[0]))
        ##for j in range(len(result[0])):
            ##result[0][j] = feature_map.shape[2]/(input_tensor.shape[2]/result[0][j]) 
            ##result[1][j] = feature_map.shape[2]/(input_tensor.shape[2]/result[1][j]) 

        #print("RESULT AFTER\n", result)    

        ##new_result = [[], []]
        #print(len(new_result[0]))
        #print(len(result[0]))


        #print("geuaaaaa")

        ##for j in range(len(result[0])):
            ##if j == 0:
                ##new_result[0].append(result[0][j])
                ##new_result[1].append(result[1][j])
        
            ##for k in range(len(new_result[0])):
                ##if result[0][j] == new_result[0][k] and result[1][j] == new_result[1][k]:
                    ##break
                ##else:
                    ##if k == (len(new_result[0])-1):
                        ##new_result[0].append(result[0][j])
                        ##new_result[1].append(result[1][j])
               
                    

        #print("NEW RESULT IS",new_result)
        ##extracted_locations = np.empty((len(new_result[0]), feature_map.shape[1]), float)

        ##for j in range(len(new_result[0])):
            ##extracted_locations[j] = feature_map[:, :, new_result[0][j] ,new_result[1][j]].cpu().detach().numpy()

        #print(extracted_locations)
        #print(extracted_locations.shape) 

        ##dict[rgb_input_path[i]] = extracted_locations
        #print(counter)


            
        #gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
        #gb = gb_model(input_tensor, target_category=None)

        #cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        #cam_gb = deprocess_image(cam_mask * gb)
        #gb = deprocess_image(gb)

        #cv2.imwrite(f'{"gradcam"}{counter}{"-"}{i}_cam.jpg', cam_image)
        #cv2.imwrite(f'{"gradcam"}{counter}{"-"}{i}_cam2.jpg', cam2_image)
        #cv2.imwrite(f'{"gradcam"}{counter}{"-"}{i}_cam3.jpg', cam3_image)
        #cv2.imwrite(f'{"gradcam"}{counter}{"-"}{i}_cam4.jpg', cam4_image)
        #cv2.imwrite(f'{"gradcam"}{counter}{"-"}{i}_gb.jpg', gb)
        #cv2.imwrite(f'{"gradcam"}{counter}{"-"}{i}_cam_gb.jpg', cam_gb)    

    counter = counter + 1

#print("RESULT BEFORE\n", result)
##np.save('locations_dictionary.npy', dict)
##np.save('feature_maps.npy', feature_map)

#print("Shape of tinput is", input_tensor.shape)
#feature_map = model.feature_for(input_tensor)
#print("Shape is",feature_map.shape)

#for i in range(len(result[0])):
    #result[0][i] = feature_map.shape[2]/(input_tensor.shape[2]/result[0][i]) 
    #result[1][i] = feature_map.shape[2]/(input_tensor.shape[2]/result[1][i]) 


#print("RESULT AFTER\n", result)

#extracted_locations = np.empty((len(result[0]), 2048), float)

#for i in range(len(result[0])):
    #extracted_locations[i] = feature_map[:, :, result[0][i] ,result[1][i]].cpu().detach().numpy()

#print(extracted_locations)
#print(extracted_locations.shape) 


