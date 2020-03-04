import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow import linalg



from MILR.Layers.activationLayer import activationLayer
from MILR.Layers import biasLayer
from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

import math
import numpy as np
from enum import Enum
from random import seed
from random import randint
from random import random
from datetime import datetime
from itertools import zip_longest

class denseLayer(layerNode):

	def __init__(self, layer, prev = None, next = None):
		super(denseLayer,self).__init__(layer, prev = prev, next = next)
		#config = layer.get_config()
		#self.units = config['units']
		#Common
		#self.hasBias = config['use_bias']
		#self.activationFunc = config['activation']

	def milrHandler(self, recoveryDirective):
		#check to see if checkpoint
		#go proper direction, passing back the needed data
		# reocover weights if needed
		pass

	def forwardPass(self, inputs):
		inputs = math_ops.cast(inputs, self.Tlayer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputs, self.Tlayer.kernel)
		if layer.use_bias:
			outputs = biasLayer.forwardPass(outputs, layer.bias)

		return activationLayer.forwardPass(outputs, layer.activation)

	def kernelSolver(self, inputs, outputs):
		assert self.store is not None, "Nothing Stored"

		m = inputs.shape[0]
		n = inputs.shape[1]
		p = outputs.shape[1]

		mPad, pPad = self.densePadding(m,n,p)

		if self.padded != DN.NONE:
			inputs = tf.concat([inputs, self.seededRandomTensor((mPad-m,n))],0)
			outputs = tf.concat([outputs,self.store[0]], 0)

		assert np.allclose(self.manIn, inputs,  atol=1e-08), "Input differs after padding"
		assert np.allclose(self.manOut, outputs,  atol=1e-08), "Output differs after padding"

		print(inputs.dtype)
		print(outputs.dtype)

		return linalg.lstsq(inputs, outputs, fast=False)
		#return np.linalg.lstsq(inputs, outputs,rcond=-1)[0]
		#return  np.linalg.solve(inputs,outputs)
		#return linalg.solve(inputs, outputs)
		
	def backwardPass(self, outputs):
		pass

	def layerInitilizer(self, inputData, status):
		self.rawIn = inputData
		self.rawKernel = self.Tlayer.kernel

		inputData = math_ops.cast(inputData, self.Tlayer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputData, self.Tlayer.kernel)
		#outputs = linalg.matmul(inputData,self.Tlayer.kernel)
		#outputs = np.matmul(inputData, self.Tlayer.kernel[:])

		self.rawOut = outputs

		layer = self.Tlayer
		assert len(inputData.shape) == 2, "Error: Dense Input Not 2D"
		assert len(layer.kernel.shape) == 2, "Error: Dense Kernel Not 2D"

		m = inputData.shape[0]
		n = inputData.shape[1]
		p = layer.kernel.shape[1]
		mPad, pPad = self.densePadding(m,n,p)

		MP = m * p
		NP = n * p
		weightCost = mPad * p - MP
		inputCost = mPad*pPad - MP
		checkpointCost = m * n

		if status == STAT.NO_INV:
			self.padded = DN.NONE
			status =  STAT.REQ_INV
			pPad = p
		else:
			if (checkpointCost + weightCost) < inputCost:
				self.checkpoint(inputData)
				self.padded = DN.NONE
				pPad = p
			else:
				self.checkpointed = False
				self.padded = DN.INPUTPAD

		if self.padded == DN.NONE:
			if weightCost < NP:
				self.padded = DN.WEIGHTPAD

		kernel = layer.kernel[:]

		inputData = self.padder2D(inputData,mPad-m,n,0)
		self.manIn = inputData
		kernel = self.padder2D(layer.kernel,n, pPad - p, 1)
		self.manKernel = kernel

		inputData = math_ops.cast(inputData, self.Tlayer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputData, kernel)
		#outputs = linalg.matmul(inputData,kernel)
		#outputs = tf.tensordot(inputData,kernel, 1)
		#outputs = np.matmul(inputData, kernel)
		self.manOut = outputs[:,:p]
		

		if self.padded == DN.WEIGHTPAD:
			self.store = [outputs[m:,:p]]
			outputs = outputs[:m,:p]
		elif self.padded == DN.INPUTPAD:
			self.store = [outputs[m:,:p], out[:,p:]]
			outputs = outputs[:m,:p]
		else:
			self.store = None

		assert np.allclose(self.rawOut, outputs,  atol=1e-08), "out wrong"

		if layer.use_bias:
			outputs, status = biasLayer.layerInitilizer(outputs, layer.bias, status)

		return activationLayer.staticInitilizer(outputs, layer.activation, status)


	def densePadding(self, m,n,p):
		if m < n:
			mPad = n
		else:
			mPad = m
		if p < n:
			pPad = n
		else:
			pPad = p
		return mPad, pPad

class DN(Enum):
	NONE = -1

	# Both, mPad and pPad
	INPUTPAD = 0
		
	# mPad only
	WEIGHTPAD = 1

