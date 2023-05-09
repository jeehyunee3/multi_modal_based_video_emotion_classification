# -*- coding: utf-8 -*-

# Load Packages
import cv2
import numpy as np
from pathlib import Path
import pandas as pd

# Load data information
total_csv = pd.read_csv("./data/labels/data_labels_total.csv")

train_images = []
val_images = []
test_images = []

train_labels = []
val_labels = []
test_labels = []

data_list = [[train_images, train_labels], [val_images, val_labels], [test_images, test_labels]]

# Load image path
target_path = "./data"
paths = list(Path(target_path).glob("**/27.png"))

# Convert image to numpy array
for i in range(0, len(paths)):
    img_path = str(paths[i])
    print(i, img_path[:img_path.rfind("/")])
    timelines_id = img_path[img_path.rfind("_") + 1 : img_path.rfind("/")]
    timelines_id = int(timelines_id)
    # data_id : 0(train), 1(valid), 2(test)
    data_id = int(total_csv[total_csv['timelines_id'] == timelines_id].data_type)
    imgs = []
    for j in range(0, 28):
        img = cv2.imread(img_path[:img_path.rfind("/")+1]+str(j)+".png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28,28))
        imgs.append(img)
        emotion = int(total_csv[total_csv['timelines_id'] == timelines_id].emotion) - 1
    data_list[data_id][0].append(imgs)
    data_list[data_id][1].append([emotion])

# Check result
test_videos = np.array(test_images)
print(test_videos.shape)
test_labels = np.array(test_labels)
print(test_labels.shape)

train_images = np.array(train_images)
val_images = np.array(val_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

# save numpy array dictionary
data_dict = {"train_images":train_images, "val_images":val_images, "test_images":test_images,
          "train_labels":train_labels, "val_labels":val_labels, "test_labels":test_labels}

np.save('./data/data_dict', data_dict)