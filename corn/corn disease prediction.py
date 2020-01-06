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


sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)


# In[ ]:





# In[3]:


#keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format='channels_last', validation_split=0.0, interpolation_order=1, dtype='float32')
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()   


# In[4]:


#dataset = datagen.flow_from_directory('D:\machine_learning_and _iot\PlantVillage')
#test_it = datagen.flow_from_directory('D:\machine_learning_and _iot\plant\Bacterial leaf blight', class_mode='binary', batch_size=64)
#l=datagen.flow_from_directory('D:\machine_learning_and _iot\plant_traiin', target_size=(897, 3081), color_mode='rgba', classes=None, class_mode='categorical', batch_size=64, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='jpg', follow_links=False, subset=None, interpolation='nearest')
test_it = datagen.flow_from_directory('D:\machine_learning_and _iot\PlantVillage\label')
PATH = os.path.join(os.path.dirname("D:\machine_learning_and _iot\PlantVillage"), 'PlantVillage')


# In[5]:


train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')


# In[6]:


train_dir = os.path.join(train_dir, 'corn')  # directory with our training cat pictures

validation_dir = os.path.join(validation_dir, 'corn')  # directory with our validation dog pictures


# In[7]:


num_corn_tr = len(os.listdir(train_dir))

num_corn_val = len(os.listdir(validation_dir))


total_train = num_corn_tr
total_val = num_corn_tr


# In[8]:


print('total training corn images:', num_corn_tr)

print('total validation corn images:', num_corn_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


# In[21]:


batch_size =2
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


# In[22]:


train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)


# In[23]:


train_data_gen = train_image_generator.flow_from_directory(batch_size=2,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')


# In[24]:


val_data_gen = validation_image_generator.flow_from_directory(batch_size=2,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')


# In[25]:


sample_training_images, _ = next(train_data_gen)


# In[26]:


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[27]:


plotImages(sample_training_images[:5])


# In[28]:


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


# In[ ]:





# In[29]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


print(total_val//batch_size)


# In[ ]:


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val//batch_size
)


# In[ ]:


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


# In[ ]:


print(history.history.keys())


# In[ ]:


image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)


# In[ ]:


train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))


# In[ ]:


augmented_images = [train_data_gen[0][0][0] for i in range(5)]


# In[ ]:


plotImages(augmented_images)
       


# In[ ]:


image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)


# In[ ]:


train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]


# In[ ]:


plotImages(augmented_images)


# In[ ]:


image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)


# In[ ]:


rain_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]


# In[ ]:


plotImages(augmented_images)


# In[ ]:


image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )


# In[ ]:


train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')


# In[ ]:


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


# In[ ]:


image_gen_val = ImageDataGenerator(rescale=1./255)


# In[ ]:


val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(4, activation='sigmoid')
])


# In[ ]:


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


# In[ ]:


model.save


# In[ ]:





# In[3]:


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

image_to_predict = load("C:\\Users\REPUBLIC OF GAMERS\\Desktop\\ashirbad.jpg")
result = model.predict(image_to_predict)
result= np.around(result,decimals=1)
result=result*100
print (result)
for i in result:
    
    for i,j in zip(label_map,i):
        print(i,j,'%')


# In[4]:


# image_to_predict= np.squeeze(image_to_predict,axis=0)
image_to_predict.shape
from matplotlib import pyplot as plt
plt.imshow(image_to_predict, interpolation='nearest')
plt.show()


# In[5]:


model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[6]:


model_new.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_new.summary()


# In[7]:


history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


# In[8]:


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


# In[9]:


model_new_1 = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[10]:


model_new_1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_new.summary()


# In[ ]:





# In[11]:


history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


# In[12]:


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


# In[13]:


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

image_to_predict = load("C:\\Users\REPUBLIC OF GAMERS\\Desktop\\images.jfif")
result = model.predict(image_to_predict)
result= np.around(result,decimals=3)
result=result*100
print (result)
for i in result:
    print(i)
    print(len(i))
    


# 

# In[14]:


score= model.evaluate_generator(test_generator, steps = math.ceil(test_samples/batch_size_test))


# In[ ]:




