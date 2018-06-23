# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 23:55:59 2018

@author: PavitrakumarPC
"""
import os
import keras
import numpy as np
import cv2
import pandas as pd
from keras import backend as K
from keras.models import model_from_json
import json
import matplotlib.pyplot as plt

K.set_image_dim_ordering('tf')

def decode_nn_res(res_vec, num_digits, num_classes, dummy_class):
    digits = np.array_split(res_vec, num_digits)
    actual_digits = np.argmax(digits,1)+1
    res = actual_digits[actual_digits!=dummy_class]
    return actual_digits, ''.join(map(str, res))

def process_labels(labels,max_digits):
    tmp = []
    for label in labels:
        vec = [int(float(x)) for x in label.split('_')]
        if len(vec) < max_digits:
            vec = vec + [11]*(max_digits-len(vec))
        tmp.append(vec)
    labels = np.array(tmp)
    tmp = []
    num_classes = 11
    for target in labels[:,...]:
        y = np.zeros((len(target), num_classes))
        y[np.arange(target.shape[0]), target-1] = 1
        tmp.append(y)
    labels = np.array(tmp)   
    return labels

root_dir = ''

#~~~~~~~~~~~~~~~~~~~~~~~~~~load data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('loading data for testing')

train_data = pd.read_hdf(os.path.join(root_dir,'data','train_data_processed.h5'),'table')

train_data = train_data[(train_data['num_digits']!=6) & (train_data['num_digits']!=5)]

train_img_size = (64,64)
train_labels = np.array(train_data[['top','left','width','height']])
tmp = []
train_images = train_data['img']
for img in train_images:
    tmp.append(cv2.normalize(cv2.cvtColor(cv2.resize(img,train_img_size), cv2.COLOR_BGR2GRAY).astype(np.float64), 0, 1, cv2.NORM_MINMAX)[...,np.newaxis])

train_images = np.array(tmp)
single_img_shape = train_images[0].shape


test_data = pd.read_hdf(os.path.join(root_dir,'data','test_data_processed.h5'),'table')

test_data = test_data[(test_data['num_digits']!=6) & (test_data['num_digits']!=5)]

test_img_size = (64,64)
test_labels = np.array(test_data[['top','left','width','height']])
tmp = []
test_images = test_data['img']
for img in test_images:
    tmp.append(cv2.normalize(cv2.cvtColor(cv2.resize(img,test_img_size), cv2.COLOR_BGR2GRAY).astype(np.float64), 0, 1, cv2.NORM_MINMAX)[...,np.newaxis])

test_images = np.array(tmp)
single_img_shape = test_images[0].shape


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~load models~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('loading cnn models')

#load digit detection model
with open(os.path.join(root_dir,'cnn_models','digit_detection_cnn_layers.json'),'r') as json_data:
    model_dict = json.load(json_data)

detect_model = model_from_json(json.dumps(model_dict))
detect_model.load_weights(os.path.join(root_dir,'cnn_models','digit_detection_cnn_weights.h5'))

#load digit classification model
with open(os.path.join(root_dir,'cnn_models','digit_classification_cnn_layers.json'),'r') as json_data:
    model_dict = json.load(json_data)

classification_model = keras.models.model_from_json(json.dumps(model_dict))
classification_model.load_weights(os.path.join(root_dir,'cnn_models','digit_classification_cnn_weights.h5'))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~do tests~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('testing using cnn model')

original_images = test_data['img']
original_boxes = np.array(test_data[['top','left','width','height']]).copy()

test_data['top'] = test_data['top']*(train_img_size[0]/test_data['img_height'])
test_data['left'] = test_data['left']*(train_img_size[1]/test_data['img_width'])
test_data['bottom'] = test_data['bottom']*(train_img_size[0]/test_data['img_height'])
test_data['right'] = test_data['right']*(train_img_size[1]/test_data['img_width'])
test_data['width'] = test_data['right'] - test_data['left']
test_data['height'] = test_data['bottom'] - test_data['top']

#if low memory when testing:
test_data['top'] = test_data['top'].astype(np.int64)
test_data['left'] = test_data['left'].astype(np.int64)
test_data['bottom'] = test_data['bottom'].astype(np.int64)
test_data['right'] = test_data['right'].astype(np.int64)
test_data['width'] = test_data['width'].astype(np.int64)
test_data['height'] = test_data['height'].astype(np.int64)

original_boxes_scaled_to_64 = np.array(test_data[['top','left','width','height']]).copy()

original_labels = test_data['labels']
num_digits = 4

digit_det_proc_img = []
for img in original_images:
    digit_det_proc_img.append(cv2.normalize(cv2.cvtColor(cv2.resize(img,train_img_size), cv2.COLOR_BGR2GRAY).astype(np.float64), 0, 1, cv2.NORM_MINMAX)[...,np.newaxis])

digit_det_proc_img = np.array(digit_det_proc_img)
box_preds = detect_model.predict(digit_det_proc_img)
print('box prediction absolute difference', np.sum(np.abs(box_preds-original_boxes_scaled_to_64))) #2584006

cut_images = []
orig_img_pred_boxes = []
for box,img,orig_box,orig_box_64 in zip(box_preds, original_images, original_boxes,original_boxes_scaled_to_64):
    #need to clip values! take care of cases where index is > img dims
    orig_pred_box = box.copy()
    scaled_box = box.copy()
    #predicted values should be similar to original_boxes_scaled_to_64
    #preidicted values are for images of 64x64 size, so we rescale it to it's original size 
    #this should now be similar to the original_boxes data
    scaled_box[0] = scaled_box[0]/float(train_img_size[0]/img.shape[0])
    scaled_box[1] = scaled_box[1]/float(train_img_size[1]/img.shape[1])
    scaled_box[2] = scaled_box[2]/float(train_img_size[1]/img.shape[1])
    scaled_box[3] = scaled_box[3]/float(train_img_size[0]/img.shape[0])
    start_row = np.clip(int(scaled_box[0]),1,img.shape[0])
    end_row = np.clip(int(scaled_box[0]+scaled_box[3]),1,img.shape[0])
    start_col = np.clip(int(scaled_box[1]),1,img.shape[1])
    end_col = np.clip(int(scaled_box[1]+scaled_box[2]),1,img.shape[1])
    orig_img_pred_boxes.append(scaled_box)
    
    if start_col-end_col==0:
        start_col -=1
    if start_row-end_row==0:
        start_row -=1

    #store only the cutouts
    img_tmp = img[start_row:end_row,start_col:end_col,...]
    img_tmp = cv2.resize(img_tmp,train_img_size)
    
    cut_images.append(img_tmp)

cut_images = np.array(cut_images)

#for i in range(10):
#    rand_index = np.random.randint(0,len(cut_images))
#    plt.imshow(cut_images[rand_index])
#    #print(rand_index+1)
#    plt.show()
        
digit_preds = classification_model.predict(cut_images)

actual_labels_encoded = process_labels(np.array(original_labels),num_digits)
actual_labels_encoded = np.array([np.concatenate(x,0) for x in actual_labels_encoded])
actual_labels_decoded = np.array([decode_nn_res(x,num_digits,11,11) for x in actual_labels_encoded])
actual_labels_decoded_digits = np.array(actual_labels_decoded[:,1])
actual_labels_decoded_OHE_digits = np.array(actual_labels_decoded[:,0])

score = np.concatenate(digit_preds, axis=1)
pred_labels_encoded = np.zeros(score.shape, dtype="int32")
pred_labels_encoded[score > 0.5] = 1
pred_labels_decoded = np.array([decode_nn_res(x,num_digits,11,11) for x in pred_labels_encoded])
pred_labels_decoded_digits = np.array(pred_labels_decoded[:,1])
pred_labels_decoded_OHE_digits = np.array(pred_labels_decoded[:,0])

print('box prediction absolute difference (64x64)', np.sum(np.abs(box_preds-original_boxes_scaled_to_64))) #2584006
print('box prediction absolute difference (orig size)', np.sum(np.abs(orig_img_pred_boxes-original_boxes))) #2517387
#so basically, when we multiply, the error magnifies... how do we fix this?...
#1. we can increase the input image size of digit detector to 256x256 or something and train a more robust classifier? or try different network?
#2. we can make the classifier more robust by generating images which are shifted, skewed a bit and re-train again [5% improvement]
print('class prediction accuracy',1-(np.sum(np.abs(pred_labels_encoded-actual_labels_encoded))/float(actual_labels_encoded.shape[0]*actual_labels_encoded.shape[1]))) #0.9650237257 
print('full digit prediction accuracy',np.sum(pred_labels_decoded_digits==actual_labels_decoded_digits)/float(actual_labels_decoded_digits.shape[0])) #0.510867901424
digit_accs = [0]*num_digits
for orig_vec,pred_vec in zip(actual_labels_decoded_OHE_digits,pred_labels_decoded_OHE_digits):
    for i in range(num_digits):
        digit_accs[i] += orig_vec[i]==pred_vec[i]

digit_accs = np.array(digit_accs)/float(len(pred_labels_decoded_OHE_digits))
print('individual digit accuracies:',digit_accs) 
#[ 0.70924537  0.65192102  0.84517067  0.9846931 ]

#~~~~~~~~~~~~~~~~~~~~~~~~~single image prediction~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('single image prediction results')

def find_box_and_predict_digit(input_img):
    num_digits = 4
    input_img_shape = input_img.shape
    train_img_size = (64,64)
    proc_input_img = np.array(cv2.normalize(cv2.cvtColor(cv2.resize(input_img,train_img_size), cv2.COLOR_BGR2GRAY).astype(np.float64), 0, 1, cv2.NORM_MINMAX)[...,np.newaxis])[np.newaxis,...]
    box_preds = detect_model.predict(proc_input_img)
    scaled_box = box_preds[0].copy()
    scaled_box[0] = scaled_box[0]/float(train_img_size[0]/input_img_shape[0])
    scaled_box[1] = scaled_box[1]/float(train_img_size[1]/input_img_shape[1])
    scaled_box[2] = scaled_box[2]/float(train_img_size[1]/input_img_shape[1])
    scaled_box[3] = scaled_box[3]/float(train_img_size[0]/input_img_shape[0])
    start_row = np.clip(int(scaled_box[0]),1,input_img_shape[0])
    end_row = np.clip(int(scaled_box[0]+scaled_box[3]),1,input_img_shape[0])
    start_col = np.clip(int(scaled_box[1]),1,input_img_shape[1])
    end_col = np.clip(int(scaled_box[1]+scaled_box[2]),1,input_img_shape[1])
    #need better logic to handle cases where the box is too thin
    if start_col-end_col==0:
        start_col -=1
    if start_row-end_row==0:
        start_row -=1
    #store only the cutouts
    digits_only = input_img[start_row:end_row,start_col:end_col,...]
    digits_only_resized = cv2.resize(digits_only,train_img_size)
    orig_img_box = input_img.copy()
    cv2.rectangle(orig_img_box,(start_col,start_row),(end_col,end_row),(0,255,0),1)
    plt.imshow(orig_img_box)
    plt.show()
    digit_pred = classification_model.predict(np.array(digits_only_resized)[np.newaxis,...])
    score = np.concatenate(digit_pred, axis=1)
    pred_labels_encoded = np.zeros(score.shape, dtype="int32")
    pred_labels_encoded[score > 0.5] = 1
    pred_labels_decoded = np.array([decode_nn_res(x,num_digits,11,11) for x in pred_labels_encoded])
    pred_labels_decoded_digits = np.array(pred_labels_decoded[:,1])
    #pred_labels_decoded_OHE_digits = np.array(pred_labels_decoded[:,0])
    final_digit = pred_labels_decoded_digits[0]
    print('Predicted digit:',final_digit)
    return final_digit

#some correctly classified IDs to try: 2604, 3727, 7141, 10045, 1648, 2458, 7638, 2887
find_box_and_predict_digit(test_data['img'][10045])
find_box_and_predict_digit(test_data['img'][1648])
find_box_and_predict_digit(test_data['img'][2458])
find_box_and_predict_digit(test_data['img'][2604])
find_box_and_predict_digit(test_data['img'][7141])
find_box_and_predict_digit(test_data['img'][7638])

#some misclassified IDs: 2532, 416, 2766, 10271, 5772, 1017, 4350, 12285
find_box_and_predict_digit(test_data['img'][1017])
find_box_and_predict_digit(test_data['img'][10271])
find_box_and_predict_digit(test_data['img'][12285])
find_box_and_predict_digit(test_data['img'][2532])
find_box_and_predict_digit(test_data['img'][4350])
find_box_and_predict_digit(test_data['img'][5772])

for i in range(10):
    rand_index = np.random.randint(0,len(test_data))
    print('test ID',rand_index)
    find_box_and_predict_digit(test_data['img'][rand_index])
