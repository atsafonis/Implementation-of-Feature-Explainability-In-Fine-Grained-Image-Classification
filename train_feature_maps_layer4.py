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
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.functional.classification import multilabel_precision
from torchmetrics.classification import MultilabelRecall
from torchmetrics.classification import MultilabelF1Score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        
        return feature_maps_tensor, vector_tensor

trainset = FeatureMapsDataset(path = '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/npy_saved_files_train/')
testset = FeatureMapsDataset(path = '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/npy_saved_files_test/')
batch_size = 200
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=8, drop_last=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=8, drop_last=False)

class BirdingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2048, 1024, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(1024, 512, 3)
        self.fc1 = nn.Linear(512, 312)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)

        return self.activation(x)

model = BirdingNet()                                             
model.to(device)

def loss_fn(output, target):

    loss = (output - target)**2
    loss = torch.where(target == 1, loss, loss*0.1)
    loss = torch.mean(loss)        

    return(loss)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 100

for epoch in range(0, num_epochs):
    # set the model in training mode
    model.train()
    totalTrainLoss = 0
    trainAccuracy = 0
    precision = 0
    totalTrainAccuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    missing_ones = 0
    total_missing_ones = 0

    # loop over the training set
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        th_output = torch.as_tensor((outputs - 0.5) > 0, dtype=torch.int32)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        totalTrainLoss += loss
        optimizer.step()       

        trainAccuracy = (th_output == labels).float().sum().item()
        trainAccuracy = 100 * trainAccuracy / len(th_output[0]) #312
        trainAccuracy = trainAccuracy / labels.size(0) #200
        totalTrainAccuracy += trainAccuracy

        ## Calculating Accuracy with torch.metrics with exactly the same results as my custom code for accuracy
        #metric = MultilabelAccuracy(num_labels=312).to(device)
        #mitsos = 100 * metric(th_output, labels).float().item()
        #total_mitsos += mitsos

        labels = labels.int()

        precision = torch.where(((th_output == labels) & (th_output == 1)), th_output, 0)
        precision = precision.float().sum().item()
        pr1 = torch.where(((th_output == labels) & (th_output == 1)), th_output, 0)
        pr1 = pr1.float().sum().item()
        pr2 = torch.where(((labels == 0) & (th_output == 1)), th_output, 0)
        pr2 = pr2.float().sum().item()
        precision = 100 * precision / (pr1 + pr2)
        total_precision += precision
        

        ## Calculating Precision with torch.metrics (not efficiently)
        #precision = 100 * multilabel_precision(th_output, labels, num_labels=312).float().item()
        #total_precision += precision

        recall = torch.where(((th_output == labels) & (th_output == 1)), th_output, 0)
        recall = recall.float().sum().item()
        re1 = torch.where(((th_output == labels) & (th_output == 1)), th_output, 0)
        re1 = re1.float().sum().item()
        re2 = torch.where(((labels == 1) & (th_output == 0)), 1, 0)
        re2 = re2.float().sum().item()
        recall = 100 * recall / (re1 + re2)
        total_recall += recall

        ## Calculating Recall with torch.metrics (not efficiently)
        #metric_re = MultilabelRecall(num_labels=312).to(device)
        #recall = 100 * metric_re(th_output, labels).float().item()
        #total_recall += recall

        f1 = 2 / ((1 / precision) + (1 / recall))
        total_f1 += f1

        ## Calculating F2 with torch.metrics (not efficiently)
        #metric_f1 = MultilabelF1Score(num_labels=312).to(device)
        #f1 = 100 * metric_f1(th_output, labels).float().item()
        #total_f1 += f1

        labels = labels.cpu().numpy()
        th_output = th_output.cpu().numpy()
        total_ones = np.count_nonzero(labels, axis = 1)
        
        for j in range(len(th_output)):
            count = np.where(np.logical_and(labels[j] == 1, th_output[j] == 0))
            count2 = np.asarray(count)
            counter = len(count2[0])
            missing_ones += 100*counter/total_ones[j]

        missing_ones = missing_ones/labels.shape[0]
        total_missing_ones += missing_ones
       
    avgTrainLoss = totalTrainLoss / len(trainloader)
    avgAccuracy = totalTrainAccuracy / len(trainloader)
    avgPrecision = total_precision/ len(trainloader)
    avgRecall = total_recall/ len(trainloader)
    avgF1 = total_f1/ len(trainloader)
    avgMissingOnes = total_missing_ones/ len(trainloader)
    print ("--->Epoch [{}/{}], Average Loss: {:.4f} Average Accuracy: {:.4f}".format(epoch + 1, num_epochs, avgTrainLoss, avgAccuracy))
    print("--->Avg Precision: ")
    print(avgPrecision)
    print("--->Avg Recall: ")
    print(avgRecall)
    print("--->Avg F1: ")
    print(avgF1)
    print("--->Missing Ones: ")
    print(avgMissingOnes)

    with torch.no_grad():
            model.eval()
            correct = 0
            total_correct = 0
            total = 0
            missing_ones = 0
            total_missing_ones = 0
            
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                th_output = torch.as_tensor((outputs - 0.5) > 0, dtype=torch.int32)
                total += labels.size(0)
                correct = (th_output == labels).float().sum().item()
                correct = 100 * correct / len(th_output[0])
                total_correct += correct

                labels = labels.cpu().numpy()
                th_output = th_output.cpu().numpy()
                total_ones = np.count_nonzero(labels, axis = 1)
        
                for j in range(len(th_output)):
                    count = np.where(np.logical_and(labels[j] == 1, th_output[j] == 0))
                    count2 = np.asarray(count)
                    counter = len(count2[0])
                    missing_ones += 100*counter/total_ones[j]

                missing_ones = missing_ones/labels.shape[0]
                total_missing_ones += missing_ones
            
            avgMissingOnes = total_missing_ones/ len(testloader)
            print("Accuracy of the network on the {} test images: {} %".format(total, total_correct / total))
            print("Missing ones in test: ")
            print(avgMissingOnes)

'''
with torch.no_grad():
    model.eval()
    correct = 0
    total_correct = 0
    total = 0
            
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        th_output = torch.as_tensor((outputs - 0.5) > 0, dtype=torch.int32)
        total += labels.size(0)
        correct = (th_output == labels).float().sum().item()
        correct = 100 * correct / len(th_output[0])
        total_correct += correct
            
    print("Accuracy of the network on the {} test images: {} %".format(total, total_correct / total))

    correct = 0
    total = 0
    total_correct = 0

    for data in trainloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        th_output = torch.as_tensor((outputs - 0.5) > 0, dtype=torch.int32)
        total += labels.size(0)
        correct = (th_output == labels).float().sum().item()
        correct = 100 * correct / len(th_output[0])
        total_correct += correct

    print("Accuracy of the network on the {} train images: {} %".format(total, total_correct / total))

'''    


