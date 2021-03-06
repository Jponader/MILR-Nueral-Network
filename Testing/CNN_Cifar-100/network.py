from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.append('../../')
from MILR.MILR import MILR

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras. datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras import backend as k
from keras.callbacks.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

# Helper libraries
import numpy as np
import h5py
import math
import time

import MILR.MILRTesting as Test


#tf.keras.backend.set_floatx('float64')

# Data PreProcessing
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
img_rows , img_cols = 32, 32

input_shape = (img_rows, img_cols, 3)


# Load Model
model= keras.models.load_model('model.h5')

# Build Model

num_category = 10
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

X_train = X_train/255
X_test = X_test/255



# inputs = keras.Input(shape=input_shape)
# x = Conv2D(96, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
# x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
# x = Dropout(0.2)(x)
# x = Conv2D(96, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
# x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
# x = Dropout(0.2)(x)
# x = Conv2D(80, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
# x = Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
# x = Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
# x = Conv2D(96, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
# x = Flatten()(x)
# x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
# x = Dropout(0.2)(x)
# output = Dense(num_category, activation='softmax')(x)

# model = keras.Model(inputs=inputs, outputs=output)

# opt =  keras.optimizers.SGD(lr=0.001, momentum=0.9)
# model.compile(optimizer=opt,
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


# #datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)

# for e in range(200):
#     print('Epoch', e)
#     batches = 0
#     for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=124):
#         model.fit(x_batch, y_batch, verbose=0)
#         batches += 1
#         if batches >= len(X_train) / 124:
#             break
#     test_loss, test_acc = model.evaluate(X_test, y_test)
#     model.save_weights('weights.h5')
#     model.save('model.h5')
#     if test_acc >= .9:
#     	break


model.summary()

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)

# Save Weights
#model.save_weights('weights.h5')

# Save Entire Model
#model.save('model.h5')

def testingFunction(X_test, y_test):
	test_loss, test_acc = model.evaluate(X_test, y_test)
	return test_acc

secureWeights = model.get_weights()

#model.summary()

milr = MILR(model)


# model.set_weights(secureWeights)

# Test.AESErrors(milr,40, [1E-4, math.sqrt(10)*1E-5, 1E-5, math.sqrt(10)*1E-6, 1E-6, math.sqrt(10)*1E-7, 1E-7], testingFunction,(X_test, y_test), "Round1")

# model.set_weights(secureWeights)

# Test.AES_ECC_Errors(milr,40, [1E-4, math.sqrt(10)*1E-5, 1E-5, math.sqrt(10)*1E-6, 1E-6, math.sqrt(10)*1E-7, 1E-7], testingFunction,(X_test, y_test), "Round1")

# model.set_weights(secureWeights)

# Test.eccMILR(milr,40, [1E-4, math.sqrt(10)*1E-5, 1E-5, math.sqrt(10)*1E-6, 1E-6, math.sqrt(10)*1E-7, 1E-7], testingFunction,(X_test, y_test), "Round1")

# model.set_weights(secureWeights)

# Test.RBERefftec(milr,40, [1E-4, math.sqrt(10)*1E-5, 1E-5, math.sqrt(10)*1E-6, 1E-6, math.sqrt(10)*1E-7, 1E-7], testingFunction,(X_test, y_test), "Round1")

model.set_weights(secureWeights)
Test.eccMILR(milr,40, [1E-3, math.sqrt(10)*1E-4], testingFunction,(X_test, y_test), "Round2")
model.set_weights(secureWeights)
Test.AES_ECC_Errors(milr,40, [1E-3, math.sqrt(10)*1E-4], testingFunction,(X_test, y_test), "Round2")