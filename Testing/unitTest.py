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
	model= keras.models.load_model('CNN_MNIST/model.h5')
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

	assert np.allclose(sol[:], layer.rawKernel[:],  atol=1e-06), "weights wrong"

def matrixMath():
	inMat = tf.convert_to_tensor(np.random.rand(256,256),  dtype= 'float32')
	weights = tf.convert_to_tensor(np.random.rand(256,256), dtype= 'float32')
	out =  gen_math_ops.mat_mul(inMat, weights)

	# Solve for Weights

	#solved = tf.linalg.lstsq( inMat, out, l2_regularizer=0.0, fast=False, name=None)
	#inMat64 = tf.cast(inMat, 'float64', name=None)
	#out64 = tf.cast(out, 'float64', name=None)



	solved = tf.linalg.solve( inMat, out, adjoint=False, name=None)
	#solved = tf.cast(solved, 'float32', name=None)

	#solved = linalg.lstsq(inMat, out, fast=False)
	print(solved)
	print(weights)

	assert np.allclose(weights, solved,  atol=1e-05), "weights wrong"

	#solve for Input

	weightsT = tf.transpose( weights, perm=None, conjugate=False, name='transpose')
	outT = tf.transpose( out, perm=None, conjugate=False, name='transpose')

	solvedInput = tf.linalg.solve( weightsT, outT, adjoint=False, name=None)
	solvedInput = tf.transpose( solvedInput, perm=None, conjugate=False, name='transpose')

	print(inMat)
	print(solvedInput)

	assert np.allclose(inMat, solvedInput,  atol=1e-05), "input wrong"

	out2 =  gen_math_ops.mat_mul(solvedInput, solved)
	assert np.allclose(out, out2,  atol=1e-05), "end wrong"

def padder2D():
	hold = tf.convert_to_tensor(np.random.rand(1,2))
	print(hold)
	out = tf.convert_to_tensor(np.random.rand(1,0))
	print(out)
	print(tf.concat([hold,out], 1))

def padding4D():
	start = tf.convert_to_tensor(np.random.rand(1,28,28,5))
	print(start.shape)
	pad1 =  tf.convert_to_tensor(np.random.rand(1,2,28,5))
	print(pad1.shape)
	paddedInput = tf.concat([start,pad1], 1)
	pad2 =  tf.convert_to_tensor(np.random.rand(1,30,2,5))
	print(pad2.shape)
	paddedInput = tf.concat([paddedInput,pad2], 2)
	print(paddedInput.shape)

def denseLayerValidation():
	pass

def main():


if __name__ == '__main__':
    main()