# -*- coding: utf-8 -*-

# Load Packages
import glob
import torch
import timm
from torchvision import models
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import TensorDataset, DataLoader

# Check GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing data
### load data information
te_label = pd.read_csv('labels/data_labels_test.csv')
video_name_list = []
label_list = []
prediction_list = []

### image to numpy array
def pre_img_loader(img_list):
    image = []
    for i in range(len(img_list)):
        img = np.array(cv2.imread(img_list[i], 1))
        image.append(img)
    image = np.array(image)
    return image

### set emotion label
def new_label(img_list):
    img_path = img_list[0]
    timelines_id = img_path[img_path.rfind("_") + 1 : img_path.rfind("/")]
    timelines_id = int(timelines_id)
    label = [int(te_label[te_label['timelines_id'] == timelines_id].emotion) - 1] * len(img_list)
    label = np.array(label)
    return label

### load image list
test_image_list = glob.glob('pre_test_image/*')

#print(test_image_list)
f = open('1130_vit_model/all_vit_image_test.txt', 'w')
best_name = ''
best_acc = 0.0


for j in range(1,40):
    ### set emotion label
    classes = ('분노', '슬픔', '불안', '상처', '당황', '기쁨', '중립')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    correct = 0
    total = 0

    # Test model - video emotion classifier

    ### ViT model
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7).to(device)

    ### VGG model
    # model = models.vgg16(pretrained=True)
    # model.classifier[6] = nn.Linear(in_features=4096, out_features=7)
    # model = model.to(device)
    
    ### load videos and preprocessing images extracted from video
    ### Evaluate the maximum label by summing the predicted probability values of all images
    model.load_state_dict(torch.load('1130_vit_model/16_check_epoch' + str(j) + '.pt'))
    print('1130_vit_model/16_check_epoch' + str(j) + '.pt')
    print('1130_vit_model/16_check_epoch' + str(j) + '.pt', file=f)
    model.eval()

    for i in range(0, len(test_image_list)):
        #print(f'{i} video name : {test_image_list[i]}')
        #print(f'{i} video name : {test_image_list[i]}', file=f)
        test_image_list_sub = glob.glob(test_image_list[i]+"/*")

        test_data = torch.tensor(pre_img_loader(test_image_list_sub), dtype=torch.float32)
        try:
            test_target = torch.tensor(new_label(test_image_list_sub))
        except:
            #print("error : no Image")
            #print("error : no Image", file=f)
            continue
        video_name_list.append(test_image_list[i])

        test_data = test_data.permute(0, 3, 1, 2)

        te_ds = TensorDataset(test_data, test_target)
        te_loader = DataLoader(te_ds, batch_size=1, shuffle=False)


        predict_sum = [0]*7
        label = -1
        prediction = -1

        for inputs, labels in te_loader:
            images, labels = inputs.to(device), labels.to(device)
            label = int(labels[0].tolist())
            model = model.to(device)
            outputs = model(images)
            outputs = outputs[0].tolist()
            for k in range(0, 7):
                predict_sum[k] += outputs[k]

        total += 1
        prediction = predict_sum.index(max(predict_sum))
        if label == prediction:
            correct += 1
            correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1
        #print(f'label : {label}, prediction : {prediction}')
        #print(f'label : {label}, prediction : {prediction}', file=f)
        label_list.append(label)
        prediction_list.append(prediction)
    ### Evaluate accuracy
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %', file=f)

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %', file=f)

    if best_acc < 100 * correct // total:
        best_acc = 100 * correct // total
        best_name = '1130_vit_model/16_check_epoch'+str(j)+'.pt'
    
    ### Save result
    # df = pd.DataFrame({"video_name":video_name_list, "label":label_list, "prediction":prediction_list})
    # df.to_csv("1128_model/vit_image_test(epoch2).csv", encoding="utf-8-sig", index=False)

print(f'best model :{best_name}({best_acc})')
print(f'best model :{best_name}({best_acc})', file=f)

f.close()