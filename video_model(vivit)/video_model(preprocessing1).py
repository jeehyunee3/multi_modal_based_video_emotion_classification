# -*- coding: utf-8 -*-
# Load Packages
import cv2
from pathlib import Path
import pandas as pd
import math
import os

f = open('./data/save_images.txt', 'w')
print("VIViT_ERROR_IMAGES ", file=f)

# Load data information
total_csv = pd.read_csv("./data/labels/data_labels_total.csv")
data_type_list = ["train", "valid", "test"]

# Set image save path
target_path = "./data/splited_video"
paths = list(Path(target_path).glob("**/*.mp4"))

# Extract 28 images per video
for i in range(0, len(paths)):
    video_path = str(paths[i])
    timelines_id = video_path[video_path.rfind("_") + 1:].replace(".mp4", "")
    timelines_id = int(timelines_id)
    # data_id : 0(train), 1(valid), 2(test)
    data_id = int(total_csv[total_csv['timelines_id'] == timelines_id].data_type)
    video_cap = cv2.VideoCapture(video_path)
    Frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    Frame_count = math.trunc(Frame_count / 28)

    count = 0
    while (video_cap.isOpened()):
        if count > 27 : break
        ret, image = video_cap.read()
        pre_file_path = "data/" + str(data_type_list[data_id]) + "/" + video_path[43:].replace(".mp4", "")
        try:
            if not os.path.exists(pre_file_path):
                os.makedirs(pre_file_path)
        except OSError:
            print('Error: Creating directory. ' + pre_file_path)

        try:
            if (int(video_cap.get(1)) % Frame_count == 0):  # save 28 Image
                # example) "유튜브_기타_1234_0_511989.png", "유튜브_기타_1234_1_511989.png"
                # data_id : train folder(0), valid folder(1), test folder(2)
                cv2.imwrite(pre_file_path + "/" + str(count) + ".png", image)
                count += 1
        except :
            break

    if count == 28 : #no error
        print(i, video_path[:], count)
    else : #error
        print("***error***", i, video_path[:], count)
        print(video_path[:], count, file=f)
    video_cap.release()
f.close()