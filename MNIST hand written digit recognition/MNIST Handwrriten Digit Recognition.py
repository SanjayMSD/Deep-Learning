#!/usr/bin/env python
# coding: utf-8

# ## IN THIS PROJECT WE ARE MAKING A MODEL THAT CAN PREDICT HAND WRITTEN DIGITS OR NUMBERS.

# ## [1] LODING LIBRARIES

# In[2]:


# IMPORTING LIBRARIES AND DATA SET

import tensorflow
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

tensorflow.compat.v1.disable_eager_execution()

# In[3]:


# TOTAL IMAGES CONTAINS 70K IMAGES (60K IN TRAINING & 10K IN TESTING)
# 10 NUMBERS FROM 0 TO 9

mnist.load_data()


# In[4]:


# SPLITING DATA SET INTO TRAIN AND TEST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# PRINT SHAPE OF AN 1ST IMAGE
x_train[0].shape


# ## [2] HERE WE ARE PLOTTING SAMPLE IMAGE OF OUR MNIST DATA SET

# In[5]:


# PLOTTING LIBRARY
import matplotlib.pyplot as plt
%matplotlib inline

# PICK 1ST IMAGE TO PLOT
sample = 0
image = x_train[sample]

# PLOTTING 1ST IMAGE
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()


# ## [3] DATA PREPROCESSING

# In[6]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

num_classes = 10

# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_train
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
y_test 

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"


# ### [4] CREATING CONVOLUTION MODEL

# In[9]:


input_shape = (28, 28, 1)
batch_size = 128
num_classes = 10
epochs = 30

#tensorflow.executing_eagerly()44

#tensorflow.compat.v1.enable_eager_execution


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
    
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),metrics=['accuracy'])


# In[10]:

model.summary()

hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")
model.save('mnist.h5')
print("Saving the model as mnist.h5")


# ### [5] EVALUTE MODEL 

# In[40]:


score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

