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


input_shape = (32,32, 3)



# Load Model
#model= keras.models.load_model('model.h5')

inputs = keras.Input(shape=input_shape)
x = Conv2D(64, (3, 3),activation='relu', padding='same')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3),activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3),activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(256, (3, 3),activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3),activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3),activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
#x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save Weights
#model.save_weights('weights.h5')

# Save Entire Model
#model.save('model.h5')

#model= keras.models.load_model('model.h5')

model.summary()

milr = MILR(model)

