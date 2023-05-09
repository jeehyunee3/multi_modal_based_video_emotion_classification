# -*- coding: utf-8 -*-

#Import Pakages
import os
import os.path as osp
import gc
import glob
import json
import time
import timm
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch as optims
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchmetrics import F1Score
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Remove GPU Cache
gc.collect()
torch.cuda.empty_cache()

#Fix Randomseed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

#Set GPU Usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda")

#Set Hyperparameters
batch_size = 60
num_epochs = 10000
size = (224, 224)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

#Create a TextFile(ViT)
###save the training results
f = open('./data/vit_result_'+str(batch_size)+'.txt', 'w')

'''
#Create a TextFile(VGG)
###save the training results
f = open('./data/vit_result_'+str(batch_size)+'.txt', 'w')
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


###1.3Function to generate the path of the image data in the folder
def make_datapath_list(phase='train'):
    rootpath = './data/datas/audio_image_sr_'
    target_path = rootpath + phase + '/'
    paths = list(Path(target_path).glob("**/*.png"))
    return paths;

###1.4.1 Generate the path of the audio image
train_list = make_datapath_list(phase='train')
val_list = make_datapath_list(phase='valid')
test_list = make_datapath_list(phase='test')

###1.4.2 Load the audio image framework
train_label = pd.read_csv('./data/labels/data_labels_train.csv')
val_label = pd.read_csv('./data/labels/data_labels_valid.csv')
test_label = pd.read_csv('./data/labels/data_labels_test.csv')

###1.4.3 Create and execute the data processing object
train_dataset = CustomDataset(file_list=train_list, label=train_label, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = CustomDataset(file_list=val_list, label=val_label, transform=ImageTransform(size, mean, std), phase='valid')
test_dataset = CustomDataset(file_list=test_list, label=test_label, transform=ImageTransform(size, mean, std), phase='test')

###1.5.1 Create data loader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

###1.5.2 Organize the data loader in a dictionary variable.
dataloaders_dict = {'train': train_dataloader, 'valid': val_dataloader, 'test': test_dataloader}

#2.Model Training

###2.1.1 Loading the model(ViT)
net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7).to(device)

"""
###2.1.1 Loading the model(VGG)
net = models.vgg16(pretrained=True)
net.classifier[6] = nn.Linear(in_features=4096, out_features=7)
net = net.to(device)

"""

###2.1.2 Setting the model in training mode
net.train()

###2.2.1 Setting the loss function
criterion = nn.CrossEntropyLoss()

###2.2.2 Applying optimization technique
optimizer = optim.AdamW(params=net.parameters(), lr=1e-4)

###2.2.3 Creating the F1 score object
F1Score = F1Score(7).to(device)

###2.3.1 Early stopping class
class EarlyStopping:
    def __init__(self, patience=1, verbose=False, delta=0.001, path='./checkpoint', check_epoch=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, check_epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, check_epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, check_epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, check_epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving epoch' + str(check_epoch) + ' model ...')

        ### checkpoint save (ViT)
        torch.save(model.state_dict(), './data/vit_audio_models_' + str(batch_size) + '/check_epoch' + str(check_epoch) + '.pt')

        '''
        ### checkpoint save (VGG)
        torch.save(model.state_dict(), './data/vgg_audio_models_' + str(batch_size) + '/check_epoch' + str(check_epoch) + '.pt')
        '''

        self.val_loss_min = val_loss



###2.3.2 Creating early stopping object
early_stopping = EarlyStopping(patience=30, verbose=True)

###2.4.1 Creating a list for score calculation during training
loss_list=[0]
acc_list=[0]
f1_list=[0]
mi_f1_list=[0]
mi_recall_list=[0]
mi_precision_list=[0]
ma_f1_list=[0]
ma_recall_list=[0]
ma_precision_list=[0]

vloss_list=[]
vacc_list=[]
vf1_list=[]
vmi_f1_list=[]
vmi_recall_list=[]
vmi_precision_list=[]
vma_f1_list=[]
vma_recall_list=[]
vma_precision_list=[]

###2.4.2 Training and validation function
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, device=device):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/ {num_epochs}')
        print('*' * 30)

        loss = 0.0
        acc = 0.0
        f1 = 0.0
        start = time.time()

        for phase in ['train', 'valid']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            epoch_f1 = 0.0

            mi_epoch_f1_2 = 0.0
            mi_recall = 0.0
            mi_precision = 0.0

            ma_epoch_f1_2 = 0.0
            ma_recall = 0.0
            ma_precision = 0.0
            counter = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)
                epoch_f1 += F1Score(labels.data, preds)

                labels_cpu = labels.clone().cpu().numpy()
                preds_cpu = preds.clone().cpu()

                mi_epoch_f1_2 += f1_score(labels_cpu.data, preds_cpu, average='micro')
                mi_recall += recall_score(labels_cpu.data, preds_cpu, average='micro')
                mi_precision += precision_score(labels_cpu.data, preds_cpu, average='micro')

                ma_epoch_f1_2 += f1_score(labels_cpu.data, preds_cpu, average='macro')
                ma_recall += recall_score(labels_cpu.data, preds_cpu, average='macro')
                ma_precision += precision_score(labels_cpu.data, preds_cpu, average='macro')

                counter = counter + 1

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            loss = epoch_loss
            acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            f1 = epoch_f1 / counter

            mi_f1_2 = mi_epoch_f1_2 / counter
            mi_precision_result = mi_precision / counter
            mi_recall_result = mi_recall / counter

            ma_f1_2 = ma_epoch_f1_2 / counter
            ma_precision_result = ma_precision / counter
            ma_recall_result = ma_recall / counter

            end = time.time()

            print(f'epoch : {epoch} : {phase} Loss {loss:.4f} Acc : {acc:.4f} F1-score : {f1:.4f} time : {end - start}')
            print(f'epoch : {epoch} : {phase} micro f1_2 {mi_f1_2:.4f} precision : {mi_precision_result:.4f} recall : {mi_recall_result:.4f} counter : {counter} / {len(dataloaders_dict[phase].dataset)}')
            print(f'epoch : {epoch} : {phase} macro f1_2 {ma_f1_2:.4f} precision : {ma_precision_result:.4f} recall : {ma_recall_result:.4f} counter : {counter} / {len(dataloaders_dict[phase].dataset)}')
            print(f'epoch : {epoch} : {phase} Loss {loss:.4f} Acc : {acc:.4f} F1-score : {f1:.4f} time : {end - start}', file=f)
            print(f'epoch : {epoch} : {phase} micro f1_2 {mi_f1_2:.4f} precision : {mi_precision_result:.4f} recall : {mi_recall_result:.4f} counter : {counter} / {len(dataloaders_dict[phase].dataset)}', file=f)
            print(f'epoch : {epoch} : {phase} macro f1_2 {ma_f1_2:.4f} precision : {ma_precision_result:.4f} recall : {ma_recall_result:.4f} counter : {counter} / {len(dataloaders_dict[phase].dataset)}', file=f)

        early_stopping(loss, net, check_epoch=epoch + 1)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if phase == 'train':
            loss_list.append(loss)
            acc_list.append(acc.clone().cpu().numpy())
            f1_list.append(f1.clone().cpu().numpy())
            mi_f1_list.append(mi_f1_2)
            mi_recall_list.append(mi_recall_result)
            mi_precision_list.append(mi_precision_result)
            ma_f1_list.append(ma_f1_2)
            ma_recall_list.append(ma_recall_result)
            ma_precision_list.append(ma_precision_result)

        elif phase == 'valid':
            vloss_list.append(loss)
            vacc_list.append(acc.clone().cpu().numpy())
            vf1_list.append(f1.clone().cpu().numpy())
            vmi_f1_list.append(mi_f1_2)
            vmi_recall_list.append(mi_recall_result)
            vmi_precision_list.append(mi_precision_result)
            vma_f1_list.append(ma_f1_2)
            vma_recall_list.append(ma_recall_result)
            vma_precision_list.append(ma_precision_result)


###2.4.3 Creating and executing train_model object
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, device=device)

#3. Training Result Graph
### Creating and displaying loss, acc, f1, recall, precision graph

plt.plot(loss_list, "o", c="red", markersize=4)
plt.plot(vloss_list, "x", c="blue", markersize=4)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('./data/graph/loss_'+str(batch_size)+'.png')
plt.cla()

plt.plot(acc_list, "o", c="red", markersize=4)
plt.plot(vacc_list, "x", c="blue", markersize=4)
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('./data/graph/acc_'+str(batch_size)+'.png')
plt.cla()

plt.plot(f1_list, "o", c="blue", markersize=4)
plt.plot(ma_f1_list, "o", c="cyan", markersize=4)
plt.plot(mi_f1_list, "o", c="navy", markersize=4)
plt.plot(vf1_list, "x", c="green", markersize=4)
plt.plot(vma_f1_list, "x", c="lime", markersize=4)
plt.plot(vmi_f1_list, "x", c="olive", markersize=4)
plt.ylabel('f1-score')
plt.xlabel('epoch')
plt.legend(['train', 'train_ma', 'train_mi', 'valid', 'valid_ma', 'valid_mi'], loc='upper left')
plt.savefig('./data/graph/f1_'+str(batch_size)+'.png')
plt.cla()

plt.plot(ma_recall_list, "o", c="orange", markersize=4)
plt.plot(mi_recall_list, "o", c="coral", markersize=4)
plt.plot(vma_recall_list, "x", c="yellow", markersize=4)
plt.plot(vmi_recall_list, "x", c="gold", markersize=4)
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train_ma', 'train_mi', 'valid_ma', 'valid_mi'], loc='upper left')
plt.savefig('./data/graph/recall_'+str(batch_size)+'.png')
plt.cla()

plt.plot(ma_precision_list, "o", c="orange", markersize=4)
plt.plot(mi_precision_list, "o", c="coral", markersize=4)
plt.plot(vma_precision_list, "x", c="yellow", markersize=4)
plt.plot(vmi_precision_list, "x", c="gold", markersize=4)
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train_ma', 'train_mi','valid_ma', 'valid_mi'], loc='upper left')
plt.savefig('./data/graph/precision_'+str(batch_size)+'.png')
plt.cla()


#4. Class Accuracy
###Checking the number of correct answers for each class and displaying the classification accuracy
classes = ('분노', '슬픔', '불안', '상처', '당황', '기쁨', '중립')

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

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %', file=f)

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %', file=f)

f.close()