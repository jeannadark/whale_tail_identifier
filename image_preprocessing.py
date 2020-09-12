#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

# In[2]:


from train_valid_split import train_valid_split
from train_valid_split import train_valid_dict_generator


# In[3]:


from PIL import Image
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


# In[4]:


train_labels = pd.read_csv("df_train.csv")


# In[5]:


train, valid = train_valid_split(train_labels)


# In[6]:


train_df, valid_df = train_valid_dict_generator(train,valid,train_labels)

# transform the above imported dictionaries into dataframes
train_df_final = pd.DataFrame(train_df.items(), columns=['Image', 'Id'])
valid_df_final = pd.DataFrame(valid_df.items(), columns=['Image', 'Id'])

# ### Apply Transformations

# In[7]:

def datagens():

    train_datagen = ImageDataGenerator(featurewise_center = False,
                            samplewise_center = False,
                            featurewise_std_normalization = False,
                            samplewise_std_normalization = False,
                            zca_whitening = False,
                            rotation_range = 10,
                            zoom_range = 0.1,
                            width_shift_range = 0.1,
                            height_shift_range = 0.1,
                            horizontal_flip = True,
                            vertical_flip = True)

    # In[8]:


    valid_datagen = ImageDataGenerator(featurewise_center = False,
                            samplewise_center = False,
                            featurewise_std_normalization = False,
                            samplewise_std_normalization = False,
                            zca_whitening = False,
                            rotation_range = 10,
                            zoom_range = 0.1,
                            width_shift_range = 0.1,
                            height_shift_range = 0.1,
                            horizontal_flip = True,
                            vertical_flip = True)

    return train_datagen, valid_datagen

# In[9]:

def prepareImages(df, shape, path):

    z_train = np.zeros((shape, 60, 60, 3))
    count = 0

    for fig in df['Image']:

        #load images into images of size 100x100x3
        img = image.load_img(path + fig, target_size=(60, 60, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        z_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1

    z_train = z_train / 255.0

    return z_train

def prepareLabels(df, number_of_classes):

    y_train = df["Id"]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    y_train = to_categorical(y_train, num_classes = number_of_classes)

    return y_train