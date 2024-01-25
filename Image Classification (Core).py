#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf

# Set the seed for NumPy
np.random.seed(42)

# Set the seed for TensorFlow
tf.random.set_seed(42)

import os, glob
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
tf.__version__


# In[2]:


import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# In[9]:


import os

data_dir = "C:\\data_folder"

# Get the list of files and subdirectories in the "farm_insects" folder
contents = os.listdir(data_dir)

# Print the contents
print(contents)


# In[10]:


# Getting list of img file paths (no folders)
img_files = glob.glob(data_dir+"**/*")
len(img_files)



# In[6]:


# Gettting the list of folders from data dir
subfolders = os.listdir(data_dir)
subfolders



# In[11]:


import glob

# Getting list of img file paths (ONLY, make it recursive to include subdirectories)
img_files = glob.glob(data_dir + "/**/*", recursive=True)
len(img_files)


# In[12]:


# Take a look at the first 5 filepaths
img_files[0:5]



# In[13]:


# Gettting the list of folders from data dir
subfolders = os.listdir(data_dir)
subfolders


# In[16]:


# Saving image params as vars for reuse
batch_size = 32
img_height = 96
img_width = 96



# In[17]:


# make the dataset from the main folder of images
ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
    shuffle=True,
    label_mode='categorical',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
ds



# In[18]:


# Determine number of batches in dataset
ds_size = len(ds)
ds_size



# In[19]:


# taking a sample batch to see batch shape
example_batch_imgs,example_batch_y= ds.take(1).get_single_element()
example_batch_imgs.shape



# In[20]:


# Preview y for first 5 of first batch
example_batch_y[0:5]



# In[21]:


# checking the class names
class_names = ds.class_names
class_names



# In[22]:


# Saving # of classes for reuse
num_classes = len(class_names)
num_classes



# In[23]:


# Saving dictionary of integer:string labels
class_dict = dict(zip(range(num_classes), class_names))
class_dict



# In[24]:


# Individual image shape
input_shape = example_batch_imgs[0].shape
input_shape



# In[25]:


# Set the ratio of the train, validation, test split
split_train = 0.7
split_val = 0.2
split_test = .1 
# Calculate the number of batches for training and validation data 
n_train_batches =  int(ds_size * split_train)
n_val_batches = int(ds_size * split_val)
print(f"Use {n_train_batches} batches as training data")
print(f"Use {n_val_batches} batches as validation data")
print(f"The remaining {len(ds)- (n_train_batches+n_val_batches)} batches will be used as test data.")



# In[26]:


# Use .take to slice out the number of batches 
train_ds = ds.take(n_train_batches)
# Confirm the length of the training set
len(train_ds)



# In[27]:


# Skipover the training batches
val_ds = ds.skip(n_train_batches)
# Take the correct number of validation batches
val_ds = val_ds.take(n_val_batches)
# Confirm the length of the validation set
len(val_ds)



# In[28]:


# Skip over all of the training + validation batches
test_ds = ds.skip(n_train_batches + n_val_batches)
# Confirm the length of the testing data
len(test_ds)



# In[29]:


# Determine number of batches in dataset
ds_size = len(ds)
ds_size


# ## Preview the Data and Save the Shape
# 
# 

# In[30]:


# checking the class names
class_names = ds.class_names

class_dict = dict(zip(range(len(class_names)), class_names))
class_dict


# In[31]:


# Batch Size
batch_size


# In[32]:


# taking a sample banch to see batch shape
example_batch_imgs,example_batch_y= train_ds.take(1).get_single_element()
example_batch_imgs.shape


# In[33]:


# individual image shape
# individual image shape
input_shape = example_batch_imgs[0].shape
input_shape


# In[34]:


array_to_img(example_batch_imgs[0])


# In[35]:


[*input_shape]


# ## Modeling
# 

# In[ ]:




