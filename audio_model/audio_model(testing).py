# -*- coding: utf-8 -*-
#Import Pakages
import os
import timm
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import warnings
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader

#Set GPU Usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda")

#Set Hyperparameters
size = (224,224)
classes = ('분노', '슬픔', '불안', '상처', '당황', '기쁨', '중립')

#Create a TextFile(ViT)
###save the training results
result_text_path = './data/script/vit_result_test_result.txt'
f = open(result_text_path, 'w')

'''
#Create a TextFile(VGG)
###save the training results
result_text_path = './data/script/vgg_result_test_result.txt'
f = open(result_text_path, 'w')
'''
#1. Data Loading and Processing

###1.1 Image preprocessing class
###Resize the image and convert it to a tensor
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
            ]),
            'valid': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
            ])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

###1.2 Data Loading class
###Load the image and perform preprocessing.
class CustomDataset(data.Dataset):

    def __init__(self, file_list, label, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.label = label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img_path = str(img_path)
        img = Image.open(img_path).convert('RGB')
        timelines_id = img_path[img_path.rfind("_") + 1:].replace(".png", "")
        timelines_id = int(timelines_id)
        img_transformed = self.transform(img, self.phase)
        return img_transformed, int(self.label[self.label['timelines_id'] == timelines_id].emotion) - 1


###1.3 Setting up the image data path.
test_audio_folder_path = "./data/audio_test/"
test_label_path = "./labels/data_labels_test.csv"

###1.4.1 Generate the path of the audio image
test_list = list(Path(test_audio_folder_path).glob("**/*.png"))

###1.4.2 Load the audio image framework
test_label = pd.read_csv(test_label_path)

###1.4.3 Create and execute the data processing object
test_dataset = CustomDataset(file_list=test_list, label=test_label, transform=ImageTransform(size), phase='test')

###1.4.4 Create data loader
test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False)


#2. checkpoint test

###2.1.1 Setting the checkpoint path(ViT)
checkpoint_path = './data/vit_audio/'
checkpoint_path_list = list(Path(checkpoint_path).glob("**/*.pt"))

'''
###2.1.1 Setting the checkpoint path(VGG)
checkpoint_path = './data/vgg_audio/'
checkpoint_path_list = list(Path(checkpoint_path).glob("**/*.pt"))
'''

###2.1.2 Declaring variables for checking the best checkpoint
best_accuracy = 0
best_model_path = ""

###2.2 Testing for each checkpoint
###print overall accuracy and class-wise accuracy
for path in checkpoint_path_list :
    path = str(path)

    ###Load the checkpoint(ViT)
    net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7).to(device)
    net.load_state_dict(torch.load(path))
    net.eval()

    ''' 
    ###Load the checkpoint(VGG)
    net = models.vgg16(pretrained=True)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=7).to(device)
    net.load_state_dict(torch.load(path))
    net.eval()
    '''

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    correct = 0
    total = 0

    for data in test_dataloader:
        images, labels = data
        net = net.cpu()
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    print(path)
    print(path, file=f)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %', file=f)

    if ( 100 * correct // total ) > best_accuracy :
        best_accuracy = ( 100 * correct // total )
        best_model_path = path

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %', file=f)
    print("")

###2.3 Print the best checkpoint
print("best model :: " + str(best_model_path) + " ( "+ str(best_accuracy)+" )")
print("best model :: " + str(best_model_path) + " ( "+ str(best_accuracy)+" )", file = f)

f.close()