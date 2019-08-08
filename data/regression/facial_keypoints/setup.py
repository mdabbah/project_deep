#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from skimage.io import imshow, imsave


# In[2]:


training_file = './training.csv'
trainig_data = pd.read_csv(training_file)


# In[3]:


def get_image_by_row(row, pd_dataframe):
    """
    returns 96x96 array of the image at row 'row'
    """
    image = pd_dataframe['Image'].iloc[row].split(' ')
    image = np.array([int(pix) for pix in image], dtype=np.uint8).reshape((96,96))
    return image


# In[4]:


example = get_image_by_row(0, trainig_data)
example


# In[5]:


imsave('example.png', example)


# In[6]:


trainig_data.shape


# In[7]:


trainig_data.dropna().shape


# In[8]:


os.makedirs('train', exist_ok=1)
os.makedirs('valid', exist_ok=1)
os.makedirs('test', exist_ok=1)


# In[9]:


trainig_data = trainig_data.dropna()
trainig_data = trainig_data.sample(frac=1).reset_index(drop=True)
num_training_samples = int(trainig_data.shape[0]*0.9)
valid_data = trainig_data.iloc[num_training_samples:, :]
valid_data.to_csv('valid.csv', index=False)
trainig_data = trainig_data.iloc[:num_training_samples, :]
trainig_data.to_csv('train.csv', index=False)


# In[10]:


print(f'valid samples: {valid_data.shape[0]}  training samples: {trainig_data.shape[0]}')


# In[11]:


valid_data


# In[12]:


valid_data_r = pd.read_csv('./valid.csv')
valid_data_r


# In[13]:


trainig_data_r = pd.read_csv('./train.csv')
trainig_data_r


# In[14]:


for row in range(trainig_data_r.shape[0]):
    image = get_image_by_row(row, trainig_data_r)
    imsave(f'./train/{row}.png', image)


# In[15]:


for row in range(valid_data_r.shape[0]):
    image = get_image_by_row(row, valid_data_r)
    imsave(f'./valid/{row}.png', image)


# In[16]:


test_file = './test.csv'
test_data = pd.read_csv(test_file)


# In[17]:


test_data.shape


# In[18]:


test_data.dropna().shape


# In[19]:


for row in range(test_data.shape[0]):
    image = get_image_by_row(row, test_data)
    imsave(f'./test/{row}.png', image)


# In[ ]:




