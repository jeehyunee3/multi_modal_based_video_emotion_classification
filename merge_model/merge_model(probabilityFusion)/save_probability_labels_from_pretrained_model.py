# Load Packages
import torch
import os
import torch.nn as nn
import timm
from torchvision import models, transforms
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import time
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import glob
import torch.nn.functional as F
import torch.optim as optim
import timm
import h5py



### Use GPU for calculate
device = torch.device("cuda")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())



# 1. Load model and weight
root_path = "./"

### 1.1. load image model(ViT)
image_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7).to(device)
### 1.2. load audio model(ViT)
audio_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7).to(device)

"""
### 1.1. Load image model(VGG)
image_model = models.vgg16(pretrained=True)
image_model.classifier[6] = nn.Linear(in_features=4096, out_features=7)

### 1.2. Load audio model(VGG)
audio_model = models.vgg16(pretrained=True)
audio_model.classifier[6] = nn.Linear(in_features=4096, out_features=7)

"""

### 1.3. Load weight
image_model.load_state_dict(torch.load(root_path+'models/image_epoch15.pt'))
audio_model.load_state_dict(torch.load(root_path+'models/audio_epoch33.pt'))

image_model = image_model.to(device)
image_model.eval()
audio_model = audio_model.to(device)
audio_model.eval()




# 2. Laod label and data
## 2.1. Load data
size = (224, 224)
def make_datapath_list(phase='train'):
    rootpath = root_path + 'data/audio_image_'
    target_path = rootpath + phase + '/'
    paths = list(Path(target_path).glob("**/*.png"))
    return paths

def make_dict(pathList):
    dict = {}
    for path in pathList:
        path = str(path)
        timelines_id = path[path.rfind("_") + 1:].replace(".png", "")
        dict[timelines_id] = path
    return dict


train_audio_list = make_datapath_list(phase='train')
valid_audio_list = make_datapath_list(phase='valid')
test_audio_list = make_datapath_list(phase='test')

train_image_list = glob.glob(root_path+'images/train/*')
valid_image_list = glob.glob(root_path+'images/valid/*')
test_image_list = glob.glob(root_path+'images/test/*')

train_image_dict = make_dict(train_image_list)
valid_image_dict = make_dict(valid_image_list)
test_image_dict = make_dict(test_image_list)

train_audio_dict = make_dict(train_audio_list)
valid_audio_dict = make_dict(valid_audio_list)
test_audio_dict = make_dict(test_audio_list)

### 2.2. Load Labels
train_label = pd.read_csv(root_path+'labels/data_labels_train.csv')
val_label = pd.read_csv(root_path+'labels/data_labels_valid.csv')
test_label = pd.read_csv(root_path+'labels/data_labels_test.csv')

### 2.3. Image pre-processing
class ImageTransform():
    def __init__(self, resize):
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



# 3. Extract and save probability labels
### 3.1. Declare z-score normalization furnction
def z_score_normalize(lis):
    normalized = []
    for value in lis:
        normalized_num = (value - np.mean(lis)/np.std(lis))
        normalized.append(normalized_num)
    return normalized

### 3.2. Declare function for Extracting and saving probability labels
def extract_image_features(label, video_dir_dict, audio_dict, out_path, phase):

    audioTransform = ImageTransform(size)
    
    with h5py.File(out_path, "w") as wf:
        for timeline_id in label['timelines_id']:
            ### 3.2.1. Get paths
            video_dir_path = video_dir_dict[str(timeline_id)]
            audio_path = audio_dict[str(timeline_id)]
            
            ### 3.2.2. Load and process the nth audio image
            audio_img = Image.open(audio_path).convert('RGB')
            audio_img_transformed = audioTransform(audio_img, phase)
            audio_img_transformed = audio_img_transformed.unsqueeze(0)
            audio_output = audio_model(audio_img_transformed)
            audio_output = audio_output.cpu()
            audio_output = z_score_normalize(audio_output.detach().numpy())
            audio_output = torch.tensor(audio_output, dtype=torch.float32)

            ### 3.2.3. Load and process the nth video images
            ### Get images
            image_paths = glob.glob(video_dir_path+'/*')
            ### Set default torch
            image_output = torch.empty(1, 7)

            ### Get average probability labels from images
            for i in range(len(image_paths)):
                img = np.array(cv2.imread(image_paths[i], 1))
                img = torch.tensor(img, dtype=torch.float32)
                img = img.permute(2, 0 ,1)
                img = img.unsqueeze(0)
                img = img.to(device)
                output = image_model(img)
                output = output.cpu()
                image_output = image_output + output
            image_output = image_output/len(image_paths)
            image_output = z_score_normalize(image_output.detach().numpy())
            image_output = torch.tensor(image_output, dtype=torch.float32)
            
            output = torch.cat([image_output, audio_output], dim=1)
            output = output.detach()

            ### 3.2.4. Save result by h5 format
            wf.create_dataset(str(timeline_id), data=output)            

### 3.3. saving probability label for each dataset
extract_image_features(train_label, train_image_dict, train_audio_dict, root_path+"vit_best_train_pre_result.h5", "train")
extract_image_features(val_label, valid_image_dict, valid_audio_dict, root_path+"vit_best_valid_pre_result.h5", "valid")
extract_image_features(test_label, test_image_dict, test_audio_dict, root_path+"vit_best_test_pre_result.h5", "test")