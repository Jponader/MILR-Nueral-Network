import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow import linalg



from MILR.Layers.activationLayer import activationLayer
from MILR.Layers.biasLayer import forwardPass as biasLayer
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

	def forwardPass(self, inputs, status):
		inputs = math_ops.cast(inputs, self.Tlayer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputs, self.Tlayer.kernel)
		if layer.use_bias:
			outputs, status = biasLayer(outputs, layer.bias, status)

		return activationLayer.forwardPass(outputs, layer.activation, status)

	def layerInitilizer(self, inputData, status):
		inputData = math_ops.cast(inputData, self.Tlayer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputData, self.Tlayer.kernel)

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
			else:
				self.padded = DN.STORED
				mPad = m

		if mPad != m:
			mIn = self.seededRandomTensor((mPad-m,n))
		if pPad != p:
			pIn = self.seededRandomTensor((n, pPad - p))
		
		if self.padded == DN.NONE:
			self.store = None
		elif self.padded == DN.STORED:
			self.store = layer.kernel
		else:
			mOut = gen_math_ops.mat_mul(mIn, self.Tlayer.kernel)
			if self.padded == DN.WEIGHTPAD:
				self.store = (mOut)
			else:
				self.store = (mOut, gen_math_ops.mat_mul(inputData, pIn), gen_math_ops.mat_mul(mIn, pIn))

		if layer.use_bias:
			outputs, status = biasLayer(outputs, layer.bias, status)

		return activationLayer.forwardPass(outputs, layer.activation, status)


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

	def seededRandomTensor(self, shape):
		np.random.seed(self.seeder())
		return tf.convert_to_tensor(np.random.rand(*shape),  dtype= self.Tlayer.dtype)

	def padder2D(self,inputs, x, y, axis):
		out = self.seededRandomTensor((x,y))
		return tf.concat([inputs,out], axis)


class DN(Enum):
	NONE = -1

	# Both, mPad and pPad
	INPUTPAD = 0
		
	# mPad only
	WEIGHTPAD = 1

	# Stored Plain Weights
	STORED = 2
