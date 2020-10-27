from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.append('../../')
from MILR.MILR import MILR
import MILR.MILRTesting as Test

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

import time


# Saving Checkpoints while training
import os
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

#tf.keras.backend.set_floatx('float64')

# Data PreProcessing
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
img_rows , img_cols = 32, 32

input_shape = (img_rows, img_cols, 3)


# Load Model
model= keras.models.load_model('model3.h5')

# Build Model

num_category = 10
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

X_train = X_train/255
X_test = X_test/255



# inputs = keras.Input(shape=input_shape)
# x = Conv2D(32, (3, 3),activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
# x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.2)(x)
# x = Conv2D(64, (3, 3),activation='relu', kernel_initializer='he_uniform', padding='same')(x)
# x = Conv2D(64, (3, 3),activation='relu', kernel_initializer='he_uniform', padding='same')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.2)(x)
# x = Conv2D(128, (3, 3),activation='relu', kernel_initializer='he_uniform', padding='same')(x)
# x = Conv2D(128, (3, 3),activation='relu', kernel_initializer='he_uniform', padding='same')(x)
# x = Conv2D(128, (3, 3),activation='relu', padding='same')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.2)(x)
# x = Flatten()(x)
# #x = Dense(512, activation='relu')(x)
# x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
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

# #it_train = datagen.flow(X_train, y_train, batch_size=64)
# 	# fit model

# #history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=100, validation_data=(X_test, y_test), verbose=0)

# #model.fit(it_train, epochs=100, validation_data=(X_test, y_test))

# for e in range(200):
#     print('Epoch', e)
#     batches = 0
#     for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=124):
#         model.fit(x_batch, y_batch, verbose=0)
#         batches += 1
#         if batches >= len(X_train) / 124:
#             break
#     test_loss, test_acc = model.evaluate(X_test, y_test)
#     model.save_weights('weights3.h5')
#     model.save('model3.h5')
#     if test_acc >= .9:
#     	break

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)

# # Save Weights
# model.save_weights('weights3.h5')

# # Save Entire Model
# model.save('model3.h5')






def testingFunction(X_test, y_test):
	test_loss, test_acc = model.evaluate(X_test, y_test)
	return test_acc

secureWeights = model.get_weights()

#model.summary()

milr = MILR(model)
model.summary()


#model.set_weights(secureWeights)
# def RBERefftec(self,rounds, error_Rate, testFunc, TestingData, testNumber)
#milr.RBERefftec(40, [1E-4, 5E-5, 1E-5, 5E-6, 1E-6, 5E-7, 1E-7], testingFunction,(X_test, y_test), 65)
#milr.RBERefftec(40, [1E-4, 5E-5, 1E-5, 5E-6, 1E-6, 5E-7, 1E-7], testingFunction,(X_test, y_test), "Round1")

model.set_weights(secureWeights)

Test.AESErrors(milr,40, [1E-4, 5E-5, 1E-5, 5E-6, 1E-6, 5E-7, 1E-7], testingFunction,(X_test, y_test), "Round1")
# def RBERefftecWhole(self,rounds, error_Rate, testFunc, TestingData, testNumber)
#milr.RBERefftec(40, [1E-4, 5E-5, 1E-5, 5E-6, 1E-6, 5E-7, 1E-7], testingFunction,(X_test, y_test), 65)
# milr.RBERefftecWhole(40, [1E-3, 5E-4, 1E-4, 5E-5, 1E-5, 5E-6, 1E-6, 5E-7, 1E-7], testingFunction,(X_test, y_test), "Round1")

#model.set_weights(secureWeights)
# def eccMILR(self,rounds, error_Rate, testFunc, TestingData, testNumber)
#milr.eccMILR(40, [1E-4, 5E-5, 1E-5, 5E-6, 1E-6, 5E-7, 1E-7], testingFunction,(X_test, y_test), "Round1")


# def continousRecoveryTest(self,rounds, error_Rate, testFunc, TestingData, testNumber)
#milr.continousRecoveryTest(40, [1E-5,1.5E-5,1E-6,1.5E-6,1E-7,1.5E-7], testingFunction, (X_test, y_test), 1)

#def v(self,rounds, error_Rate, testFunc, TestingData, testNumber)
#milr.LayerSpecefic(50, [1], testingFunction, (X_test, y_test), "Round1")


