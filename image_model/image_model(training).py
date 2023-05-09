# -*- coding: utf-8 -*-

# Load Packages
import pandas as pd
import cv2
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
import glob
import torch
import torch.nn as nn
from torchmetrics import F1Score
import time
import torch.optim as optim
import random
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Set random seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Check GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

# Preprocessing data
### load data information
tr_label = pd.read_csv('./data/labels/data_labels_train.csv')
v_label = pd.read_csv('./data/labels/data_labels_valid.csv')
te_label = pd.read_csv('./data/labels/data_labels_test.csv')

train_image_list = glob.glob('./data/pre_train_image/*')
val_image_list = glob.glob('./data/pre_val_image/*')
test_image_list = glob.glob('./data/pre_test_image/*')

### Extract video into (224,224) images per frame
def split_video(t_list, mode):
    path = (t_list.replace('./data/splited_video/유튜브_기타_', '')).replace('.mp4', '')
    pre_file_path = "pre_" + mode + "_image/" + path
    vidcap = cv2.VideoCapture(t_list)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    count, count1 = 0, 0
    while (vidcap.isOpened()):
        try:
            if not os.path.exists(pre_file_path):
                print(pre_file_path)
                os.makedirs(pre_file_path)
        except OSError:
            print('Error: Creating directory. ' + pre_file_path)
        ret, image = vidcap.read()
        if image is None: break
        if (int(vidcap.get(1)) % fps == 0):
            re_image = cv2.resize(image, (224, 224))
            # print('Saved frame number : ' + str(int(vidcap.get(1))))
            cv2.imwrite("%s/%d.jpg" % (pre_file_path, count), re_image)
            # print('Saved frame%d.jpg' % count)
            count += 1
        count1 += 1
        if int(vidcap.get(1)) < count1:
            break
    vidcap.release()


### rename data
def rename(file):
    new_file = []
    for i in range(len(file)):
        new_file = np.append(new_file, (
                    './data/splited_video/' + file["file_name"][i][:-4] + "_" + str(file["timelines_id"][i]) + ".mp4"))
    return new_file


### set data label
def label(list, mode):
    new_label = []
    if mode == "train": Mode, k = tr_label, 22
    if mode == "val": Mode, k = v_label, 20
    if mode == "test": Mode, k = te_label, 21
    for i in range(len(list)):
        emotion = (Mode['emotion'])[((np.where(Mode['timelines_id'] == int((list[i])[k:])))[0])[0]]
        fold_num = glob.glob(list[i] + '/*')
        for _ in range(len(fold_num)):
            new_label.append(emotion - 1)
    return new_label

### load image
def pre_img_loader(img_list):
    image = []
    for i in range(len(img_list)):
        a = glob.glob(img_list[i] + '/*')
        for img_dir in a:
            img = np.array(cv2.imread(img_dir, 1))
            image.append(img)
    image = np.array(image)
    return image


### set data name
# train_list = rename(tr_label)
# val_list = rename(v_label)
# test_list = rename(te_label)

# for i in range (len(train_list)):
#     split_video(train_list[i], "train")
# for i in range (len(val_list)):
#     split_video(val_list[i], "val")
# for i in range (len(test_list)):
#     split_video(test_list[i], "test")

### set data label
# train_label = np.array(label(train_image_list, "train"))
# val_label = np.array(label(val_image_list, "val"))
# test_label = np.array(label(test_image_list, "test"))

### save data information
# np.savetxt('pre_train_label.csv', train_label, delimiter=',')
# np.savetxt('pre_val_label.csv', val_label, delimiter=',')
# np.savetxt('pre_test_label.csv', test_label, delimiter=',')

### Convert image to Tensor
train_data = torch.tensor(pre_img_loader(train_image_list), dtype=torch.float32)
val_data = torch.tensor(pre_img_loader(val_image_list), dtype=torch.float32)
test_data = torch.tensor(pre_img_loader(test_image_list), dtype=torch.float32)
train_target = torch.tensor((np.loadtxt('pre_train_label.csv', delimiter=',')), dtype=torch.long)
val_target = torch.tensor((np.loadtxt('pre_val_label.csv', delimiter=',')), dtype=torch.long)
test_target = torch.tensor((np.loadtxt('pre_test_label.csv', delimiter=',')), dtype=torch.long)

### Change data dimensions [N, H, W, C] -> [N, C, H, W] 
train_data = train_data.permute(0, 3, 1, 2)
val_data = val_data.permute(0, 3, 1, 2)
test_data = test_data.permute(0, 3, 1, 2)

### create dataset
tr_ds = TensorDataset(train_data, train_target)
va_ds = TensorDataset(val_data, val_target)
te_ds = TensorDataset(test_data, test_target)

### create dataloader
batch_size = 16
tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=True)
te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

f = open('./data/model/vit_result.txt', 'w')

### dataloader to dictionary
dataloaders_dict = {'train': tr_loader, 'valid': va_loader, 'test': te_loader}

### check the operation
batch_iterator = iter(dataloaders_dict['train'])
inputs, labels = next(batch_iterator)

# ViT Training
### load pretrained model ViT
import timm
net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7).to(device)

# VGG Training
### load pretrained model vgg
# net = models.vgg16(pretrained=True)
### Set final output unit to 7
# net.classifier[6] = nn.Linear(in_features=4096, out_features=7)
# net = net.to(device)

### set train mode
net.train()

### set cross entropy loss function
criterion = nn.CrossEntropyLoss()

### set optimizer adam
import torch as optims
optimizer = optim.AdamW(params=net.parameters(), lr=1e-4)


### early stopping function
class EarlyStopping:
    def __init__(self, patience=1, verbose=False, delta=0.001, path='./checkpoint', check_epoch=0):
        """
        Args:
            patience (int) : patience
            verbose (bool): True - improvement message
            delta (float): Minimum change in monitered quality
            path (str): checkpoint save path
        """
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
            print('')
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving epoch' + str(
                check_epoch) + ' model ...')
            print('')
        # torch.save(model.state_dict(),
        #            './data/model/'+str(batch_size)+'_check_epoch' + str(check_epoch) + '.pt')
        self.val_loss_min = val_loss


early_stopping = EarlyStopping(patience=30, verbose=True)
F1Score = F1Score(7).to(device)

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

### Model training function
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, device=device):
    for epoch in range(num_epochs):
        print()
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
            print()
            print(f'epoch {epoch+1} : {phase} Loss {loss:.4f} Acc : {acc:.4f} F1-score : {f1:.4f} time : {end - start}')
            print(
                f'epoch {epoch+1} : {phase} micro f1_2 {mi_f1_2:.4f} precision : {mi_precision_result:.4f} recall : {mi_recall_result:.4f} counter : {counter} / {len(dataloaders_dict[phase].dataset)}')
            print(
                f'epoch {epoch+1} : {phase} macro f1_2 {ma_f1_2:.4f} precision : {ma_precision_result:.4f} recall : {ma_recall_result:.4f} counter : {counter} / {len(dataloaders_dict[phase].dataset)}')
            print(f'epoch {epoch+1} : {phase} Loss {loss:.4f} Acc : {acc:.4f} F1-score : {f1:.4f} time : {end - start}',
                  file=f)
            print(
                f'epoch {epoch+1} : {phase} micro f1_2 {mi_f1_2:.4f} precision : {mi_precision_result:.4f} recall : {mi_recall_result:.4f} counter : {counter} / {len(dataloaders_dict[phase].dataset)}',
                file=f)
            print(
                f'epoch {epoch+1} : {phase} macro f1_2 {ma_f1_2:.4f} precision : {ma_precision_result:.4f} recall : {ma_recall_result:.4f} counter : {counter} / {len(dataloaders_dict[phase].dataset)}',
                file=f)
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

        torch.save(net.state_dict(), '1128_model/' + str(batch_size) + '_check_epoch' + str(epoch + 1) + '.pt')
        early_stopping(loss, net, check_epoch=epoch + 1)

        if early_stopping.early_stop:
            print("Early stopping")
            break

### train model
num_epochs = 10000
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, device=device)

# Drawing graph of loss, accuracy, f1-scorer, recall and precision
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

classes = ('분노', '슬픔', '불안', '상처', '당황', '기쁨', '중립')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

correct = 0
total = 0

# Test model
for inputs, labels in te_loader:
    images, labels = inputs.to(device), labels.to(device)
    net = net.to(device)
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

# Evaluate accuracy
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %', file=f)

f.close()