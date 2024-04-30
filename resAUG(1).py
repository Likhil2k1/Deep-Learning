#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pathlib


# In[3]:


import os
data_path = pathlib.Path(r"C:\Users\Likhil\Desktop\adata")


# In[4]:


train_ds = tf.keras.utils.image_dataset_from_directory(
 data_path,
 validation_split=0.2,
 subset="training",
 seed=123,
 image_size=(256,256),
 batch_size=32)


val_ds = tf.keras.utils.image_dataset_from_directory(
 data_path,
 validation_split=0.2,
 subset="validation",
 seed=123,
 image_size=(256,256),
 batch_size=32)


# In[5]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")


# In[6]:


from tensorflow.keras import layers
resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(256,256),
  layers.Rescaling(1./255)
])


# In[7]:


resnet_model = Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(256,256,3),
                   pooling='avg',classes=2,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)


# In[8]:


from tensorflow.keras.layers import Dropout
resnet_model.add(Flatten())
resnet_model.add(Dense(4096, activation='relu'))
resnet_model.add(Dropout(0.2))

resnet_model.add(Dense(1, activation='sigmoid'))

resnet_model.summary()


# In[9]:


from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate=0.001)
resnet_model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

early_stop= EarlyStopping(monitor='val_accuracy',patience=10,verbose=1,mode='auto')

mcp_save = ModelCheckpoint('mdl_resnt50_1.hdf5', save_best_only=True, mode='auto',period=1)



history=resnet_model.fit(train_ds,
          epochs=5,
          batch_size=32,
          validation_data=(val_ds), 
          callbacks=[early_stop,mcp_save])


# In[14]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


# In[11]:


plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




