import sys
sys.path.append('../')
from MILR.MILR import MILR
import MILR.Layers as M
import MILR.status as STAT

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

import numpy as np


def addLayerTesting():
	inputs = keras.Input(shape=(4,4))
	x1 = keras.layers.Flatten()(inputs)
	x2 = keras.layers.Flatten()(x1)
	add = keras.layers.Add()([x1,x2])
	output = keras.layers.Dense(10, activation=tf.nn.softmax)(add)
	model = keras.Model(inputs=inputs, outputs=output)
	model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
	model.summary()

	milr = MILR(model)
	return model


def activationFucntion():
	vector = tf.convert_to_tensor([[25,10,15,20,30]], dtype= np.float32)
	print(vector)
	result = M.activationLayer.forwardPass(vector, 'linear')
	print(result)


def concatPractice():
	inputs = tf.convert_to_tensor(np.random.rand(2,3))
	out = tf.convert_to_tensor(np.random.rand(2,3))

	print(inputs)
	print()
	print(out)
	print(0)

	print(tf.concat([inputs,out], 1))

def main():
	model = concatPractice()

if __name__ == '__main__':
    main()