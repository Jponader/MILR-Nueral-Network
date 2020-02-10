from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.append('../../')
from MILR.MILR import MILR

import tensorflow as tf
from tensorflow import keras
import h5py
import time
import os



checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)


# Data PreProcessing
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
							 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
test_length = 10000
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load Model
model= keras.models.load_model('model.h5')

"""
model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28, 28)),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(10, activation=tf.nn.softmax)
])
"""
"""
inputs = keras.Input(shape=(28,28))
x = keras.layers.Flatten()(inputs)
y = keras.layers.Dense(128, activation=tf.nn.relu)(x)
output = keras.layers.Dense(10, activation=tf.nn.softmax)(y)
model = keras.Model(inputs=inputs, outputs=output)

model.run_eagerly = True

model.compile(optimizer='adam',
							loss='sparse_categorical_crossentropy',
							metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, callbacks = [cp_callback])

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
"""
# Save Weights
#model.save_weights('weights.h5')
# model.load_weights()

# Save Entire Model
#model.save('model.h5')

modelNew = keras.models.load_model('model.h5')


#model.summary()

milr = MILR(model)


