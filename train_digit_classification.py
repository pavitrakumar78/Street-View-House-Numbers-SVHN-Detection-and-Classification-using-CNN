# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:22:43 2018

@author: PavitrakumarPC
"""

import keras
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, Input, Dropout, Flatten, Activation, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.utils.np_utils import to_categorical  
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import json
from sklearn.cross_validation import train_test_split
K.set_image_dim_ordering('tf') 

def decode_nn_res(res_vec, num_digits, num_classes, dummy_class):
    digits = np.array_split(res_vec, num_digits)
    actual_digits = np.argmax(digits,1)+1
    res = actual_digits[actual_digits!=dummy_class]
    return res, ''.join(map(str, res))

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

def standardize(img):
    s = img - np.mean(img, axis=(2,0,1), keepdims=True)
    s /= (np.std(s, axis=(2,0,1), keepdims=True) + 1e-7)
    return s

root_dir = ''


train_data = pd.read_hdf(os.path.join(root_dir,'data','train_data_processed.h5'),'table')

train_data = train_data[(train_data['num_digits']!=6) & (train_data['num_digits']!=5)]

#we slightly shift each image in 4 directions to make the classifier more robust
#so, below quadruples the dataset we have
#this takes a long time to complete!
extra_data = pd.DataFrame(columns = train_data.columns)
for index, row in train_data.iterrows():
    c_row = row.copy()
    c_row['left'] = c_row['left']-5
    c_row['right'] = c_row['right']-5
    c_row['width'] = c_row['right'] - c_row['left']
    c_row['cut_img'] = c_row['img'].copy()[int(c_row['top']):int(c_row['top']+c_row['height']),int(c_row['left']):int(c_row['left']+c_row['width']),...]
    extra_data = extra_data.append(c_row, ignore_index=True)
    
    c_row = row.copy()
    c_row['left'] = c_row['left']+5
    c_row['right'] = c_row['right']+5
    c_row['width'] = c_row['right'] - c_row['left']
    c_row['cut_img'] = c_row['img'].copy()[int(c_row['top']):int(c_row['top']+c_row['height']),int(c_row['left']):int(c_row['left']+c_row['width']),...]
    extra_data = extra_data.append(c_row, ignore_index=True)
    
    c_row = row.copy()
    c_row['top'] = c_row['top']-5
    c_row['bottom'] = c_row['bottom']-5
    c_row['height'] = c_row['bottom'] - c_row['top']
    c_row['cut_img'] = c_row['img'].copy()[int(c_row['top']):int(c_row['top']+c_row['height']),int(c_row['left']):int(c_row['left']+c_row['width']),...]
    extra_data = extra_data.append(c_row, ignore_index=True)
    
    c_row = row.copy()
    c_row['top'] = c_row['top']+5
    c_row['bottom'] = c_row['bottom']+5
    c_row['height'] = c_row['bottom'] - c_row['top']
    c_row['cut_img'] = c_row['img'].copy()[int(c_row['top']):int(c_row['top']+c_row['height']),int(c_row['left']):int(c_row['left']+c_row['width']),...]
    extra_data = extra_data.append(c_row, ignore_index=True)
    if (index%1000)==0:
        print(str(index)+'/'+str(train_data.shape[0])+' done')

train_data = pd.concat([train_data,extra_data])


#training
num_digits = 4
train_img_size = (64,64)
train_labels = process_labels(np.array(train_data['labels']),num_digits) #don't use [['labels']]

tmp = []
sel_indices = []
train_images = train_data['cut_img'] #img or cut_img
for index,img in enumerate(train_images):
    #some images have very small dims because of image aug step we did previously
    if img.shape[0]>=8 and img.shape[1]>=3:
        tmp.append(cv2.resize(img,train_img_size))
        sel_indices.append(index)

train_images = np.array(tmp)
train_labels = train_labels[sel_indices,...]
#del train_data, tmp

single_img_shape = train_images[0].shape

k_size = 7
cnn3 = Input(shape=single_img_shape)
layer = Conv2D(48, k_size,k_size, border_mode='same')(cnn3)
layer = BatchNormalization()(layer)
layer = Activation('tanh')(layer)
layer = MaxPooling2D(pool_size=(2, 2), strides = 2) (layer)
layer = Dropout(0.2)(layer)
layer = Conv2D(64, k_size,k_size, border_mode='same')(cnn3)
layer = BatchNormalization()(layer)
layer = Activation('tanh')(layer)
layer = MaxPooling2D(pool_size=(2, 2), strides = 2) (layer)
layer = Dropout(0.2)(layer)
layer = Conv2D(64, k_size,k_size, border_mode='same')(cnn3)
layer = BatchNormalization()(layer)
layer = Activation('tanh')(layer)
layer = MaxPooling2D(pool_size=(2, 2), strides = 2) (layer)
layer = Dropout(0.2)(layer)
layer = Conv2D(128, k_size,k_size, border_mode='same')(layer)
layer = BatchNormalization()(layer)
layer = Activation('tanh')(layer)
layer = MaxPooling2D(pool_size=(2, 2)) (layer)
layer = Dropout(0.2)(layer)
layer = Conv2D(192, k_size,k_size, border_mode='same')(layer)
layer = BatchNormalization()(layer)
layer = Activation('tanh')(layer)
layer = MaxPooling2D(pool_size=(2, 2)) (layer)
layer = Dropout(0.2)(layer)
layer = Conv2D(256, k_size,k_size, border_mode='same')(layer)
layer = BatchNormalization()(layer)
layer = Activation('tanh')(layer)
layer = MaxPooling2D(pool_size=(2, 2)) (layer)
layer = Dropout(0.2)(layer)
layer = Conv2D(256, k_size,k_size, border_mode='same')(layer)
layer = BatchNormalization()(layer)
layer = Activation('tanh')(layer)
layer = MaxPooling2D(pool_size=(2, 2)) (layer)
layer = Dropout(0.2)(layer)
layer = Flatten()(layer)
layer = Dense(1024, activation="tanh")(layer)
layer = Dropout(0.2)(layer)

#output
d1 = Dense(11, activation='softmax')(layer)
d2 = Dense(11, activation='softmax')(layer)
d3 = Dense(11, activation='softmax')(layer)
d4 = Dense(11, activation='softmax')(layer)
model = Model(cnn3, [d1,d2,d3,d4])
optim = RMSprop(lr=0.00005)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
model.fit(train_images, [train_labels[:,ind,:] for ind in range(num_digits)], epochs=50, batch_size=64, validation_split=0.05, verbose=1, shuffle=True)

preds = model.predict(train_images[5001:5500,...])
score = np.concatenate(preds, axis=1)
round_score = np.zeros(score.shape, dtype="int32")
round_score[score > 0.5] = 1

tmp = []
for vec in train_labels[5001:5500,...]:
    tmp.append(np.concatenate(vec,0))

ac_score = np.array(tmp)

print('prediction accuracy',1-(np.sum(np.abs(round_score-ac_score))/float(round_score.shape[0]*round_score.shape[1]))) #1.0

pred_digits = np.array([decode_nn_res(x,num_digits,11,11)[1] for x in round_score])
actual_digits = np.array([decode_nn_res(x,num_digits,11,11)[1] for x in ac_score])

print('acc',np.sum(pred_digits==actual_digits)/float(pred_digits.shape[0])) #1.0

#~~~~~~~~~~~~~~~~~~~~~~~~~~testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_data = pd.read_hdf(os.path.join(root_dir,'data','test_data_processed.h5'),'table')

test_data = test_data[(test_data['num_digits']!=6) & (test_data['num_digits']!=5)]
num_digits = 4
test_img_size = (64,64)
test_labels = process_labels(np.array(test_data['labels']),num_digits) #don't use [['labels']]

tmp = []
test_images = test_data['cut_img'] #img or cut_img
for img in test_images:
    tmp.append(cv2.resize(img,test_img_size))

test_images = np.array(tmp)

preds = model.predict(test_images)
score = np.concatenate(preds, axis=1)
round_score = np.zeros(score.shape, dtype="int32")
round_score[score > 0.5] = 1

tmp = []
for vec in test_labels:
    tmp.append(np.concatenate(vec,0))

ac_score = np.array(tmp)

print('prediction accuracy',1-(np.sum(np.abs(round_score-ac_score))/float(round_score.shape[0]*round_score.shape[1]))) #0.988

pred_digits = np.array([decode_nn_res(x,num_digits,11,11)[1] for x in round_score])
actual_digits = np.array([decode_nn_res(x,num_digits,11,11)[1] for x in ac_score])

print('acc',np.sum(pred_digits==actual_digits)/float(pred_digits.shape[0])) #0.80

#~~~~~~~~~~~~~~~~~~~~~~~~~~~saving~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Architecture
with open(os.path.join(root_dir,'cnn_models','digit_classification_cnn_layers.json',"r"),"w") as f:
    f.write(model.to_json())

f.close()
# Weights
model.save_weights(os.path.join(root_dir,'cnn_models','digit_classification_cnn_weights.h5'))

#model.save(os.path.join(root_dir,'cnn_models','digit_classification_cnn_fullmodel.h5'))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
