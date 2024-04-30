#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


from keras.preprocessing.image import ImageDataGenerator
from skimage import io


# In[3]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob


# In[4]:


import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action="ignore")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
#tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout,Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping 


# In[5]:


get_ipython().system('pip install numpy')


# In[6]:


get_ipython().system('pip install pandas')


# In[7]:


get_ipython().system('pip install tensorflow')


# In[8]:


get_ipython().system('pip install seaborn')


# In[9]:


get_ipython().system('pip install sklearn')


# In[10]:


from keras.preprocessing.image import ImageDataGenerator
from skimage import io


# In[11]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob


# In[12]:


import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action="ignore")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
#tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout,Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping 


# In[13]:


get_ipython().system('pip install opencv-python')


# In[14]:


import os
os.sys.path


# In[ ]:





# In[17]:


import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array  # Assuming you're using TensorFlow 2.x

# Define your image directory (replace with the actual path)
img_dir = r'C:\Users\Leela Krishna.Divi\Desktop\yes'

# Create a directory to save augmented images (optional, but recommended)
augmented_dir = os.path.join(img_dir, "augmented")
os.makedirs(augmented_dir, exist_ok=True)  # Create directory only if it doesn't exist

# Define image augmentation parameters (adjust as needed)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    data_format="channels_last",
    brightness_range=[0.5, 1.5]
)

# Find all image files in the directory
files = glob(os.path.join(img_dir, "*"))

# Process images (choose either option 1 or 2):

# Option 1: Process individual images (more memory efficient for large datasets)
for f1 in files:
    img = cv2.imread(f1)
    x = img_to_array(img)  # Convert to NumPy array
    x = np.expand_dims(x, axis=0)  # Add a batch dimension

    # Augment the image
    batch = datagen.flow(x, batch_size=1, save_to_dir=augmented_dir, save_prefix="augmented", save_format="jpg")

    # Process the augmented image (e.g., display it)
    for augmented_img in batch:
        # ... your code to process the augmented image (e.g., display or save)
        print(f"Augmented image shape: {augmented_img.shape}")  # Example output

# Option 2: Process all images as a single batch (less memory efficient but faster)
# **Caution:** If you have a large number of images, this might cause memory issues.
# Consider using option 1 for large datasets.
data = []
for f1 in files:
    img = cv2.imread(f1)
    img_array = img_to_array(img)
    data.append(img_array)

x = np.array(data)  # Combine all images into a batch

# Augment the batch of images
batch = datagen.flow(x, batch_size=len(x), save_to_dir=augmented_dir, save_prefix="augmented", save_format="jpg")

# Process the augmented batches (loop through them)
for augmented_batch in batch:
    # ... your code to process the augmented batch (multiple images)
    print(f"Augmented batch shape: {augmented_batch.shape}")  # Example output


# In[ ]:


#IMAGE AUGMENTATION ON NO FOLDER

import keras
import cv2
import os
import glob
datagen = ImageDataGenerator(rotation_range =15, 
                         width_shift_range = 0.2, 
                         height_shift_range = 0.2,  
                         rescale=1./255, 
                         shear_range=0.2, 
                         zoom_range=0.2, 
                         horizontal_flip = True, 
                         fill_mode = 'nearest', 
                         data_format='channels_last', 
                         brightness_range=[0.5, 1.5]) 


img_dir =r"C:\Users\Leela Krishna.Divi\Desktop\no" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img) 

x = img_to_array(img)
x = x.reshape((1,) + x.shape)


i = 0
path, dirs, files = next(os.walk(r"C:\Users\Leela Krishna.Divi\Desktop\no"))
file_count = len(files) #to find number of files in folder

for batch in datagen.flow (x, batch_size=1, save_to_dir =r"C:\Users\Leela Krishna.Divi\Desktop\no",save_prefix="a",save_format='jpg'):
    i+=1
    if i==file_count:
          break


# In[ ]:


tumor_dir=r"C:\Users\Leela Krishna.Divi\Desktop\yes"
healthy_dir=r"C:\Users\Leela Krishna.Divi\Desktop\no"
filepaths = []
labels= []
dict_list = [tumor_dir, healthy_dir]
for i, j in enumerate(dict_list):
    flist=os.listdir(j)
    for f in flist:
        fpath=os.path.join(j,f)
        filepaths.append(fpath)
        if i==0:
          labels.append('tumor')
        else:
          labels.append('healthy')
print(dict_list)


# In[ ]:


Fseries = pd.Series(filepaths, name="filepaths")
Lseries = pd.Series(labels, name="labels")
tumor_data = pd.concat([Fseries,Lseries], axis=1)
tumor_df = pd.DataFrame(tumor_data)
print(tumor_df.head())
print(tumor_df["labels"].value_counts())

#shape of datatset
tumor_df.shape

train_images, test_images = train_test_split(tumor_df, test_size=0.3, random_state=42)
train_set, val_set = train_test_split(tumor_df, test_size=0.2, random_state=42)


# In[ ]:


print(train_set.shape)
print(test_images.shape)
print(val_set.shape)
print(train_images.shape)

image_gen = ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet_v2.preprocess_input)
train = image_gen.flow_from_dataframe(dataframe= train_set,x_col="filepaths",y_col="labels",
                                      target_size=(244,244),
                                      color_mode='rgb',
                                      class_mode="categorical", #used for Sequential Model
                                      batch_size=32,
                                      shuffle=False            #do not shuffle data
                                     )
test = image_gen.flow_from_dataframe(dataframe= test_images,x_col="filepaths", y_col="labels",
                                     target_size=(244,244),
                                     color_mode='rgb',
                                     class_mode="categorical",
                                     batch_size=32,
                                     shuffle= False
                                    )

val = image_gen.flow_from_dataframe(dataframe= val_set,x_col="filepaths", y_col="labels",
                                    target_size=(244,244),
                                    color_mode= 'rgb',
                                    class_mode="categorical",
                                    batch_size=32,
                                    shuffle=False
                                   )


# In[ ]:


classes=list(train.class_indices.keys())
print (classes)

def show_brain_images(image_gen):
    test_dict = test.class_indices
    classes = list(test_dict.keys())
    images, labels=next(image_gen) # get a sample batch from the generator 
    plt.figure(figsize=(20,20))
    length = len(labels)
    if length<25:
        r=length
    else:
        r=25
    for i in range(r):
        plt.subplot(5,5,i+1)
        image=(images[i]+1)/2 #scale images between 0 and 1
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color="green",fontsize=16)
        plt.axis('off')
    plt.show()
    

show_brain_images(train)


# In[ ]:


from keras.applications.vgg16 import VGG16, preprocess_input
IMG_SIZE=(224,224)
vgg16_weight_path = r"C:\Users\Leela Krishna.Divi\Downloads\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
base_model = VGG16(
    weights=vgg16_weight_path,
    include_top=False, 
    input_shape=IMG_SIZE + (3,)
)
base_model.summary()


# In[ ]:


NUM_CLASSES = 2
from keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop

model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=1e-4),
    metrics=['accuracy']
)

model.summary()


# In[ ]:


import time

start = time.time()
History = model.fit(train,validation_data= val, epochs=10,verbose=1)
print("Total time: ", time.time() - start, "seconds")


# In[ ]:


acc = History.history["accuracy"] # report of model
val_acc = History.history["val_accuracy"] # history of validation data

loss = History.history["loss"]        # Training loss
val_loss = History.history["val_loss"] # validation loss

plt.figure(figsize=(8,8))
plt.subplot(2,1,1) # 2 rows and 1 columns
#plotting respective accuracy
plt.plot(acc,label="Training Accuracy")
plt.plot(val_acc, label="Validation Acccuracy")

plt.legend()
plt.ylabel("Accuracy", fontsize=12)
plt.title("Training and Validation Accuracy", fontsize=12)


plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
 
plt.plot(loss, label="Training Loss")      #Training loss
plt.plot(val_loss, label="Validation Loss") # Validation Loss

plt.legend()
plt.ylim([min(plt.ylim()),1])
plt.ylabel("Loss", fontsize=12)
plt.title("Training and Validation Losses", fontsize=12)


# In[ ]:


model.evaluate(test, verbose=1)


# In[ ]:


pred = model.predict(test)
pred = np.argmax(pred, axis=1) #pick class with highest  probability

labels = (train.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred2 = [labels[k] for k in pred]


# In[ ]:


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

y_test = test_images.labels # set y_test to the expected output
print(classification_report(y_test, pred2))
print("Accuracy of the Model:",accuracy_score(y_test, pred2)*100,"%")

plt.figure(figsize = (10,5))
cm = confusion_matrix(y_test, pred2)
sns.heatmap(cm, annot=True, fmt = 'g')


# In[ ]:





# In[ ]:





# In[ ]:




