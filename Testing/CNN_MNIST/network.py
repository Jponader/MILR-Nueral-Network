from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.append('../../')
from MILR.MILR import MILR

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras. datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras import backend as k

# Helper libraries
import numpy as np
import h5py

import time

test_length = 10000

# Saving Checkpoints while training
import os
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)


# Data PreProcessing
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_rows , img_cols = 28, 28

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255



# Load Model
#model= keras.models.load_model('model.h5')

# Build Model
num_category = 10
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

"""
inputs = keras.Input(shape=input_shape)
#x = Conv2D(32, kernel_size=(3, 3),activation='relu', padding="SAME")(inputs)
x = Conv2D(32, kernel_size=(3, 3),activation='relu')(inputs)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_category, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2, batch_size = 128)


test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)

# Save Weights
model.save_weights('weights.h5')

# Save Entire Model
model.save('model.h5')
"""
model= keras.models.load_model('model.h5')
secureWeights = model.get_weights()

#model.summary()

milr = MILR(model)

milr.RBERefftec(10, [1E-1,1E-2,1E-3,1E-4,1E-5,1E-6,1E-7,1E-8,1E-9,1E-10], (X_test, y_test))

milr.continousRecoveryTest(20, 1E-5, (X_test, y_test), 1)
#model.set_weights(secureWeights)
#milr.continousRecoveryTest(20, 1E-6, (X_test, y_test), 2)
#model.set_weights(secureWeights)
#milr.continousRecoveryTest(20, 1E-7, (X_test, y_test), 3)
#model.set_weights(secureWeights)
#milr.continousRecoveryTest(20, 1E-8, (X_test, y_test), 4)
#model.set_weights(secureWeights)
#milr.continousRecoveryTest(20, 1E-9, (X_test, y_test), 5)

#milr.error_Sim(20, 1E-5, baseModel = model, TestingData =(X_test, y_test) )
