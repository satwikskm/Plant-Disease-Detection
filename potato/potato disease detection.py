#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib.inline
import tensorflow as tf

from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog,blob_log,blob_doh
from math import sqrt
from matplotlib import image as img
from skimage.color import rgb2gray 
import glob
from skimage.io import imread
from keras.models import Sequential 
from keras import layers
import cv2
from keras import utils
from keras.utils import Sequence
import keras
from keras import optimizers
import numpy as np
from PIL import Image
import os
from keras import models
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,Conv1D
import cv2
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:





# In[45]:


test_it = datagen.flow_from_directory('D:\machine_learning_and _iot\PlantVillage\label_2')


# In[46]:


sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='sgd')


# In[ ]:





# In[5]:


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()   


# In[6]:


PATH = os.path.join(os.path.dirname("D:\machine_learning_and _iot\PlantVillage"), 'PlantVillage')


# In[7]:


train_dir = os.path.join(PATH, 'train_1')
validation_dir = os.path.join(PATH, 'test_1')


# In[8]:


train_dir = os.path.join(train_dir, 'potato') 

validation_dir = os.path.join(validation_dir, 'potato') 


# In[9]:


num_corn_tr = len(os.listdir(train_dir))

num_corn_val = len(os.listdir(validation_dir))


total_train = num_corn_tr
total_val = num_corn_tr


# In[10]:


print('total training corn images:', num_corn_tr)

print('total validation corn images:', num_corn_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


# In[11]:


batch_size =128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


# In[29]:


train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)


# In[30]:


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')


# In[31]:


val_data_gen = validation_image_generator.flow_from_directory(batch_size=2,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')


# In[32]:


sample_training_images, _ = next(train_data_gen)


# In[33]:


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[34]:


plotImages(sample_training_images[:5])


# In[35]:


model=Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150,150,3))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten()) 
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu')) 
model.add(layers.Dense(2, activation='softmax'))
model.summary()


# In[ ]:





# In[36]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[37]:


model.compile(loss='mean_squared_error', optimizer='sgd')


# In[38]:


model.summary()


# In[39]:


print(total_val//batch_size)


# In[40]:


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // 1,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val //1,
)


# In[41]:


print(history.history.keys())


# In[ ]:





# In[42]:


acc = history.history['loss']
val_acc = history.history['val_loss']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[49]:


model.save('potato.h5')


# In[50]:


from PIL import Image
import numpy as np
from skimage import transform

def load(filename):
    np_image = Image.open(filename) #Open the image
    np_image = np.array(np_image).astype('float32')/255 #Creates a numpy array as float and divides by 255.
    np_image = transform.resize(np_image, (150, 150, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

   #picture1 prediction

label_map = (test_it.class_indices)
print (label_map)

image_to_predict = load('C:\\Users\REPUBLIC OF GAMERS\\Desktop\\pbl.jpg')
result = model.predict(image_to_predict)
result= np.around(result,decimals=3)
result=result*100
print (result)


# In[51]:


image_to_predict= np.squeeze(image_to_predict,axis=0)
image_to_predict.shape
from matplotlib import pyplot as plt
plt.imshow(image_to_predict, interpolation='nearest')
plt.show()


# In[ ]:





# In[ ]:




