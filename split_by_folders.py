#!/usr/bin/env python
# coding: utf-8

# In[1]:


from train_valid_split import train_valid_split, train_valid_dict_generator


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import os
import shutil


# In[4]:


train_labels = pd.read_csv("df_train.csv")


# In[5]:


train, valid = train_valid_split(train_labels)


# In[6]:


train_df, valid_df = train_valid_dict_generator(train,valid,train_labels)


# In[8]:


# run once to setup

out_folder_path = r'...\Documents\whale_identification\whale_identification\data\train_rev1'
in_folder_path = r'...\Documents\whale_identification\whale_identification\data\train'

for image_name in train_df.keys():
    cur_image_path = os.path.join(in_folder_path, image_name)
    cur_image_out_path = os.path.join(out_folder_path, image_name)
    shutil.copyfile(cur_image_path, cur_image_out_path)


# In[7]:


# run once to setup

out_folder_path = r'...\Documents\whale_identification\whale_identification\data\val2'
in_folder_path = r'...\Documents\whale_identification\whale_identification\data\train'

for image_name in valid_df.keys():
    cur_image_path = os.path.join(in_folder_path, image_name)
    cur_image_out_path = os.path.join(out_folder_path, image_name)
    shutil.copyfile(cur_image_path, cur_image_out_path)




