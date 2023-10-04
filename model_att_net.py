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


class ATT_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2048, 1024, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(1024, 512, 3)
        self.fc1 = nn.Linear(512, 312)

        self.activation = torch.nn.Sigmoid()
        self.shortcut = nn.Identity()

    def forward(self, x):
        x = self.shortcut(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = self.fc1(x)

        return self.activation(x)
