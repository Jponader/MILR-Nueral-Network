import sys
sys.path.append('../')
from MILR.MILR import MILR
import MILR.Layers as M
import MILR.status as STAT

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow import linalg

import numpy as np

def defualtModel():
	model= keras.models.load_model('FashionMNIST/model.h5')
	model.summary()
	milr = MILR(model)
	return milr

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
	return milr


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

def weightSolver():
	model = defualtModel()
	layer = model.milrModel[3]
	print(type(layer))

	sol = layer.kernelSolver(layer.rawIn, layer.rawOut)

	print(sol)
	print(layer.rawKernel[:])

	assert np.allclose(sol[:], layer.rawKernel[:],  atol=1e-08), "weights wrong"

def matrixMath():
	inMat = tf.convert_to_tensor(np.random.rand(784,784),  dtype= 'float64')
	weights = tf.convert_to_tensor(np.random.rand(784,128), dtype= 'float64')
	out =  gen_math_ops.mat_mul(inMat, weights)
	print(out.shape)

	solved = linalg.lstsq(inMat, out, fast=False)
	print(solved.shape)

	assert np.allclose(weights, solved,  atol=1e-08), "weights wrong"

def padder2D():
	hold = tf.convert_to_tensor(np.random.rand(1,2))
	print(hold)
	out = tf.convert_to_tensor(np.random.rand(1,0))
	print(out)
	print(tf.concat([hold,out], 1))

def main():
	weightSolver()
	#matrixMath()

if __name__ == '__main__':
    main()