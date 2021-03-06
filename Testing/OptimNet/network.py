from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.append('../../')
from MILR.MILR import MILR

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras. datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras import backend as k
from keras.callbacks.callbacks import EarlyStopping

# Helper libraries
import numpy as np
import h5py

import time


# Saving Checkpoints while training
import os
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)


# Data PreProcessing
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
img_rows , img_cols = 32, 32

input_shape = (img_rows, img_cols, 3)


# Load Model
#model= keras.models.load_model('model.h5')

# Build Model

num_category = 10
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

inputs = keras.Input(shape=input_shape)
x = Conv2D(16, (2, 2),activation='relu', padding='same')(inputs)
x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
x = Conv2D(256, (2, 2),activation='relu', padding='same')(x)
x = Conv2D(256, (1, 1),activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (2, 2),activation='relu', padding='same')(x)
x = Conv2D(64, (1, 1),activation='relu', padding='same')(x)
x = Conv2D(64, (1, 1),activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_category, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=75, batch_size = 32, validation_data=(X_test, y_test))

# test_loss, test_acc = model.evaluate(X_test, y_test)

# print('Test accuracy:', test_acc)

# Save Weights
model.save_weights('weights.h5')

# Save Entire Model
model.save('model.h5')


def testingFunction(X_test, y_test):
	test_loss, test_acc = model.evaluate(X_test, y_test)
	return test_acc

model= keras.models.load_model('model.h5')
secureWeights = model.get_weights()

#model.summary()

milr = MILR(model)
model.summary()



# def RBERefftec(self,rounds, error_Rate, testFunc, TestingData, testNumber)
# milr.RBERefftec(40, [1E-1,1.5E-1,1E-2,1.5E-2,1E-3,1.5E-3,1E-4,1.5E-4,1E-5,1.5E-5,1E-6,1.5E-6,1E-7,1.5E-7], testingFunction,(X_test, y_test), 2)
# milr.RBERefftec(40, [1E-4,1.5E-4,1E-5,1.5E-5,1E-6,1.5E-6,1E-7,1.5E-7], testingFunction,(X_test, y_test), 2)
# milr.RBERefftec(2, [1E-5], testingFunction,(X_test, y_test), 1)
# model.set_weights(secureWeights)


# model.set_weights(secureWeights)
# def continousRecoveryTest(self,rounds, error_Rate, testFunc, TestingData, testNumber)
# milr.continousRecoveryTest(40, [1E-5,1.5E-5,1E-6,1.5E-6,1E-7,1.5E-7], testingFunction, (X_test, y_test), 1)


# model.set_weights(secureWeights)
# def v(self,rounds, error_Rate, testFunc, TestingData, testNumber)
# milr.LayerSpecefic(40, [1E-4,1.5E-4,1E-5,1.5E-5,1E-6,1.5E-6,1E-7,1.5E-7], testingFunction, (X_test, y_test), 1)