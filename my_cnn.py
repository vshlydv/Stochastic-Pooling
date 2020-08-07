import numpy as np
import os
from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('leapgestrecog/leapGestRecog/00/'):
    if not j.startswith('.'): 
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
lookup

x_data = []
y_data = []
datacount = 0 
for i in range(0, 10): 
    for j in os.listdir('leapgestrecog/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): 
            count = 0 
            for k in os.listdir('leapgestrecog/leapGestRecog/0' + 
                                str(i) + '/' + j + '/'):
                img = Image.open('leapgestrecog/leapGestRecog/0' + 
                                 str(i) + '/' + j + '/' + k).convert('L')
                img = img.resize((64, 64))
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1
                if count > 10:
                    break
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1)


for i in range(0, 10):
    plt.imshow(x_data[i*20 , :, :])
    plt.title(reverselookup[y_data[i*20 ,0]])
    plt.show()


import keras
keras.backend.set_image_data_format('channels_first')
from keras.utils import to_categorical
y_data = to_categorical(y_data)

x_data = x_data.reshape((datacount, 1, 64, 64))
x_data /= 255

from sklearn.model_selection import train_test_split
x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)

from keras import layers
from keras import models
from keras.layers import Lambda
import stochastic_pooling as st

model=models.Sequential()
model.add(layers.Conv2D(32, 5, 5, activation='relu', input_shape=(1, 64, 64))) 
model.add(Lambda(st.stochastic_max_pool_x, output_shape = st.output_shape_of_lambda, arguments = {'image_shape' : (model.output_shape[2], model.output_shape[3]), 'pool_shape' : (2, 2)}))
model.add(layers.Convolution2D(64, 3, 3, activation='relu')) 
model.add(Lambda(st.stochastic_max_pool_x, output_shape = st.output_shape_of_lambda, arguments = {'image_shape' : (model.output_shape[2], model.output_shape[3]), 'pool_shape' : (2, 2)}))
model.add(layers.Convolution2D(64, 3, 3, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=1, validation_data=(x_validate, y_validate))

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy with sotchastic pooling')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss with sotchastic pooling')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()