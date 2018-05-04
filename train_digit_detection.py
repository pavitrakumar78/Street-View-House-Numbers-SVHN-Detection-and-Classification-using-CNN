# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:14:21 2018

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
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils.np_utils import to_categorical  
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import json
K.set_image_dim_ordering('tf') 

root_dir = ''

train_data = pd.read_hdf(os.path.join(root_dir,'data','train_data_processed.h5'),'table')

train_data = train_data[(train_data['num_digits']!=6) & (train_data['num_digits']!=5)]
#very less 6-digit/5-digit train samples, so we stop with 4

#training
train_img_size = (64,64)


train_data['top'] = train_data['top']*(train_img_size[0]/train_data['img_height'])
train_data['left'] = train_data['left']*(train_img_size[1]/train_data['img_width'])
train_data['bottom'] = train_data['bottom']*(train_img_size[0]/train_data['img_height'])
train_data['right'] = train_data['right']*(train_img_size[1]/train_data['img_width'])
train_data['width'] = train_data['right'] - train_data['left']
train_data['height'] = train_data['bottom'] - train_data['top']

#if low memory when testing, use below conversions:
train_data['top'] = train_data['top'].astype(np.int64)
train_data['left'] = train_data['left'].astype(np.int64)
train_data['bottom'] = train_data['bottom'].astype(np.int64)
train_data['right'] = train_data['right'].astype(np.int64)
train_data['width'] = train_data['width'].astype(np.int64)
train_data['height'] = train_data['height'].astype(np.int64)


train_labels = np.array(train_data[['top','left','width','height']])
tmp = []
train_images = train_data['img']
for img in train_images:
    tmp.append(cv2.normalize(cv2.cvtColor(cv2.resize(img,train_img_size), cv2.COLOR_BGR2GRAY).astype(np.float64), 0, 1, cv2.NORM_MINMAX)[...,np.newaxis])

train_images = np.array(tmp)
single_img_shape = train_images[0].shape


k_size = 3
cnn1 = Input(shape=single_img_shape)
#x = BatchNormalization()(cnn1)
x = Conv2D(32, k_size, k_size, activation='relu', border_mode='same')(cnn1)
x = BatchNormalization()(x)
x = Conv2D(32, k_size, k_size, activation='relu', border_mode='same')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(64, k_size, k_size, activation='relu', border_mode='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, k_size, k_size, activation='relu', border_mode='same')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(128, k_size, k_size, activation='relu', border_mode='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, k_size, k_size, activation='relu', border_mode='same')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
h = Dense(4)(x)
model = Model(input=cnn1, output=h)
model.compile(loss='mse', optimizer='adadelta')
model.fit(train_images, train_labels, epochs=50, batch_size=128, validation_split=0.2, verbose=1)
#final training loss: 0.9939 | val loss: 17.47 (val loss fluctuated between 15 and 25)

#~~~~~~~~~~~~~~~~~~~~~~~~~~testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_data = pd.read_hdf(os.path.join(root_dir,'data','test_data_processed.h5'),'table')

test_data = test_data[(test_data['num_digits']!=6) & (test_data['num_digits']!=5)]

test_img_size = (64,64)

test_labels = np.array(test_data[['top','left','width','height']])
tmp = []
test_images = test_data['img']
for img in test_images:
    tmp.append(cv2.normalize(cv2.cvtColor(cv2.resize(img,test_img_size), cv2.COLOR_BGR2GRAY).astype(np.float64), 0, 1, cv2.NORM_MINMAX)[...,np.newaxis])

test_images = np.array(tmp)

actual_labels = test_labels
preds = model.predict(test_images)

print('mean squared error difference',np.sum(np.power(actual_labels-preds,2))) #96777101.9255
print('mean error difference',np.sum(np.abs(actual_labels-preds))) #1198707.77919

#~~~~~~~~~~~~~~~~~~~~~~~~~~~saving~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Architecture
with open(os.path.join(root_dir,'cnn_models','digit_detection_cnn_layers.json'),"w") as f:
    f.write(model.to_json())

f.close()
# Weights
model.save_weights(os.path.join(root_dir,'cnn_models','digit_detection_cnn_weights.h5'))

#model.save(os.path.join(root_dir,'cnn_models','digit_detection_cnn_fullmodel.h5'))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
