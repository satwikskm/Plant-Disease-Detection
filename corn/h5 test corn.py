from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
test_it = datagen.flow_from_directory('D:\machine_learning_and _iot\PlantVillage\label')
model=load_model('corn.h5')
model.summary()
from PIL import Image
import numpy as np
from skimage import transform

def load(filename):
    np_image = Image.open(filename) 
    np_image = np.array(np_image).astype('float32')/255 
    np_image = transform.resize(np_image, (150, 150, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image
label_map = (test_it.class_indices)
print (label_map)

image_to_predict = load("C:\\Users\REPUBLIC OF GAMERS\\Desktop\\leaf.jpg")
result = model.predict(image_to_predict)
result= np.around(result,decimals=3)
result=result*100
print (result)
for i in result:
    print(i)
    print(len(i))




