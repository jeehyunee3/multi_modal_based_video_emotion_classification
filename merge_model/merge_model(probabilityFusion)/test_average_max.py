# 패키지 로드
import pandas as pd
import numpy as np



# 1. Load data
###  1.1. Load label information
test_label = pd.read_csv('./data_labels_test.csv')
### 1.2. Load probability labels datas
test_data = pd.read_csv('./preprocessed_test_df.csv')


# 2. Set variables
### 2.1. set variable for correct count
correct = 0
### 2.2. set list for by correct count of classes
correct_li = [0,0,0,0,0,0,0]



# 3. Evaluate test data accuracy based on Average Max
for timeline_id in test_label.timelines_id:
    data = test_data[test_data['timeline_id'] == timeline_id].iloc[0]

    ### Load probability labels of audio classification model
    audio_result = data[1:8]
    audio_result.index = range(7)
    ### Load probability labels of image classification model
    image_result = data[8:15]
    image_result.index = range(7)

    ### Merge probability labels
    result = audio_result + image_result
    ### Calculate predicted class with Average Max
    max_index = result.argmax()

    ### Get target value
    target = int(test_label[test_label['timelines_id'] == timeline_id].emotion) - 1

    ### If the prediction is correct
    if target == max_index:
        correct = correct + 1
        correct_li[max_index] = correct_li[max_index] + 1



# 4. Print result
### Print correct count
print("correct : ", correct)
### Print accuracy(percentage)
print("accuracy : ", correct / (len(test_label.timelines_id) - error))
### Print accuracy for each classes
print(correct_li)