import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim import lr_scheduler
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
import argparse
import imageio


parser = argparse.ArgumentParser()
parser.add_argument('--neg_weight', type=float)
args = parser.parse_args()
                    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ImagesDataset (Dataset) :
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        self.images_path = '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/datasets/CUB_200_2011/images.txt'
        self.attributes_path = '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/datasets/CUB_200_2011/attributes/image_attribute_labels.txt'
        self.attributes_labels = np.loadtxt(self.attributes_path,dtype="int",usecols =(0,1,2))
        self.images = np.loadtxt(self.images_path, delimiter=' ', dtype=str)
        self.img_value_dict = {}
        self.attributes_dict = {}
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))


        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        self.len_train_value = len(train_file_list)
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        self.len_test_value = len(test_file_list)

        if self.is_train:
            self.train_img = [os.path.join(self.root, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
        
        if not self.is_train:
            self.test_img = [os.path.join(self.root, 'images', test_file) for test_file in
                             test_file_list[:data_len]]

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

    def __getitem__(self, index):
        if self.is_train:
            img = imageio.imread(self.train_img[index])
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
        
            #height, width = img.height, img.width
            #height_scale = self.input_size / height
            #width_scale = self.input_size / width

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            imgpath = self.train_img[index]
            

            id = self.img_value_dict[imgpath.split('/')[10]]
           
            attribute_vector = self.attributes_dict[id]
            attribute_tensor = torch.from_numpy(attribute_vector)

        else:
            img = imageio.imread(self.test_img[index])
    
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
    
            #height, width = img.height, img.width
            #height_scale = self.input_size / height
            #width_scale = self.input_size / width

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            imgpath = self.test_img[index]

            id = self.img_value_dict[imgpath.split('/')[10]]
            attribute_vector = self.attributes_dict[id]
            attribute_tensor = torch.from_numpy(attribute_vector)



        #scale = torch.tensor([height_scale, width_scale])

        return img, attribute_tensor, imgpath

    def __len__(self):
        if self.is_train:
            return self.len_train_value
        else:
            return self.len_test_value
    

batch_size = 64
trainset = ImagesDataset(input_size=448, root='/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/datasets/CUB_200_2011', is_train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=8, drop_last=False)
testset = ImagesDataset(input_size=448, root='/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/datasets/CUB_200_2011', is_train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=8, drop_last=False)

model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 312)                                           
model.to(device)

def loss_fn(output, target):

    loss = (output - target)**2
    loss = torch.where(target == 1, loss, loss*(args.neg_weight))
    loss = torch.mean(loss)        

    return(loss)

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

num_epochs = 100
best_f1 = 0

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

    #exp_lr_scheduler.step()   

    avgTrainLoss = totalTrainLoss / len(trainloader)
    avgAccuracy = totalTrainAccuracy / len(trainloader)
    avgPrecision = total_precision/ len(trainloader)
    avgRecall = total_recall/ len(trainloader)
    avgF1 = total_f1/ len(trainloader)
    avgMissingOnes = total_missing_ones/ len(trainloader)
    
   
    
    with torch.no_grad():
            model.eval()
            correct = 0
            total_correct = 0
            total = 0
            missing_ones = 0
            total_missing_ones = 0

            precision = 0
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            
            
            
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                th_output = torch.as_tensor((outputs - 0.5) > 0, dtype=torch.int32)
                total += labels.size(0)
                correct = (th_output == labels).float().sum().item()
                correct = 100 * correct / len(th_output[0])
                total_correct += correct

                labels = labels.int()

                precision = torch.where(((th_output == labels) & (th_output == 1)), th_output, 0)
                precision = precision.float().sum().item()
                pr1 = torch.where(((th_output == labels) & (th_output == 1)), th_output, 0)
                pr1 = pr1.float().sum().item()
                pr2 = torch.where(((labels == 0) & (th_output == 1)), th_output, 0)
                pr2 = pr2.float().sum().item()
                precision = 100 * precision / (pr1 + pr2)
                total_precision += precision

                recall = torch.where(((th_output == labels) & (th_output == 1)), th_output, 0)
                recall = recall.float().sum().item()
                re1 = torch.where(((th_output == labels) & (th_output == 1)), th_output, 0)
                re1 = re1.float().sum().item()
                re2 = torch.where(((labels == 1) & (th_output == 0)), 1, 0)
                re2 = re2.float().sum().item()
                recall = 100 * recall / (re1 + re2)
                total_recall += recall

                f1 = 2 / ((1 / precision) + (1 / recall))
                total_f1 += f1



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
            
            test_MissingOnes = total_missing_ones/ len(testloader)
            test_Precision = total_precision/ len(testloader)
            test_Recall = total_recall/ len(testloader)
            test_F1 = total_f1/ len(testloader)
            #print("Accuracy of the network on the {} test images: {} %".format(total, total_correct / total))
            #print("Missing ones in test: ")
            #print(avgMissingOnes)
            #print ("--->Epoch [{}/{}], Average Test Accuracy: {:.4f} Average Test Precision: {:.4f} Average Test Recall: {:.4f} Average Test F1: {:.4f} Average Test Missing: {:.4f}".format(epoch + 1, num_epochs, total_correct / total, test_Precision, test_Recall, test_F1, test_MissingOnes))
    if test_F1 > best_f1:
        best_f1 = test_F1
        torch.save(model.state_dict(), '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/model_best_test_f1_images.pth')
        print("Saved")     
    print ("--->Epoch [{}/{}], Average Loss: {:.4f} Average Accuracy: {:.4f} Average Precision: {:.4f} Average Recall: {:.4f} Average F1: {:.4f} Average Missing: {:.4f} Average Test Accuracy: {:.4f} Average Test Precision: {:.4f} Average Test Recall: {:.4f} Average Test F1: {:.4f} Average Test Missing: {:.4f}".format(epoch + 1, num_epochs, avgTrainLoss, avgAccuracy, avgPrecision, avgRecall, avgF1, avgMissingOnes, total_correct / total, test_Precision, test_Recall, test_F1, test_MissingOnes))