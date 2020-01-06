#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib.inline
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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


# In[2]:


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()


# In[3]:


test_it = datagen.flow_from_directory('D:\machine_learning_and _iot\PlantVillage\label_3')
PATH = os.path.join(os.path.dirname("D:\machine_learning_and _iot\PlantVillage"), 'PlantVillage')


# In[4]:


train_dir = os.path.join(PATH, 'train_2')
validation_dir = os.path.join(PATH, 'test_2')


# In[5]:


train_dir = os.path.join(train_dir, 'rice') 

validation_dir = os.path.join(validation_dir, 'rice') 


# In[6]:


num_rice_tr = len(os.listdir(train_dir))

num_rice_val = len(os.listdir(validation_dir))


total_train = num_rice_tr
total_val = num_rice_tr


# In[7]:


print('total training paddy images:', num_rice_tr)

print('total validation paddy images:', num_rice_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


# In[8]:


batch_size =10
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


# In[9]:


train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)


# In[11]:


train_data_gen = train_image_generator.flow_from_directory(batch_size=10,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')


# In[12]:


val_data_gen = validation_image_generator.flow_from_directory(batch_size=2,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')


# In[13]:


sample_training_images, _ = next(train_data_gen)


# In[14]:


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[15]:


plotImages(sample_training_images[:5])


# In[16]:


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
model.add(layers.Dense(3, activation='softmax'))
model.summary()


# In[17]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[18]:


model.summary()


# In[20]:


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=100,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


# In[21]:


print(history.history.keys())


# In[22]:


acc = history.history['acc']
val_acc = history.history['val_acc']

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


# In[23]:


model.save('paddy.h5')


# In[30]:


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

image_to_predict = load('C:\\Users\REPUBLIC OF GAMERS\\Desktop\\bac.jpg')
result = model.predict(image_to_predict)
result= np.around(result,decimals=1)
result=result*100
print (result)


# In[23]:


image_to_predict= np.squeeze(image_to_predict,axis=0)
image_to_predict.shape
from matplotlib import pyplot as plt
plt.imshow(image_to_predict, interpolation='nearest')
plt.show()


# In[ ]:





# In[ ]:




