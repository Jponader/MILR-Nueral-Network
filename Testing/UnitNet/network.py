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


input_shape = (5, 1)



# Load Model
#model= keras.models.load_model('model.h5')

inputs = keras.Input(shape=input_shape)
x = Flatten()(inputs)
x = Dense(10, activation='relu')(x)
output = Dense(5, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save Weights
#model.save_weights('weights.h5')

# Save Entire Model
#model.save('model.h5')

#model= keras.models.load_model('model.h5')

#model.summary()

milr = MILR(model)

