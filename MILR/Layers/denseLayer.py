from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops


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


	def layerInitilizer(self, inputData, status):
		return self.forwardPass(inputData, status)

	def forwardPass(self, inputs, status):
		self.rawIn = inputs
		outputs = math_ops.cast(inputs, layer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputs, layer.kernel)
		self.rawOut = outputs
		self.rawKernel =self.Tlayer.kernel


		layer = self.Tlayer
		assert len(inputs.shape) == 2, "Error: Dense Input Not 2D"
		assert len(layer.kernel.shape) == 2, "Error: Dense Kernel Not 2D"

		m = inputs.shape[0]
		n = inputs.shape[1]
		p = layer.kernel.shape[0]
		mPad, pPad = self.densePadding(m,n,p)

		MP = m * p
		NP = n * p
		weightCost = mPad * p - MP
		inputCost = mPad*pPad - MP
		checkpointCost = m * n
		

		print("outputSize", MP)
		print("weightCost", weightCost)
		print("inputCost", inputCost)
		print("checkpointCost", checkpointCost)
		print("plainWeights", NP)

		if status == STAT.NO_INV:
			self.padded = DN.NONE
			status =  STAT.REQ_INV
		else:
			if checkpointCost < inputCost:
				self.checkpoint(inputs)
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

		if mPad != m or pPad != p:
			inputs = self.padder(inputs, mPad, pPad)
		
		outputs = math_ops.cast(inputs, layer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputs, layer.kernel)


		if self.padded == DN.INPUTPAD:
			self.store = None
			outputs = outputs
		elif self.padded == DN.WEIGHTPAD:
			self.store = None
			outputs = outputs
		elif self.padded == DN.STORED:
			self.store = layer.kernel
		else:
			self.store = None

		"""
		if status == STAT.NO_INV:
			status =  STAT.REQ_INV
		else :
			print("			Possible Checkpoint")
			invert = self.invertibility()
			checkpoint = inputs.size
			if checkpoint <= invert:
				#self.checkpoint(inputs)
				status = STAT.NO_INV
			else: 
				status= STAT.REQ_INV
				#modify code for gettting metadata
"""


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

	def padder(self,input, mPad, pPad, status):

		if self.padded == DN.INPUTPAD:
			#both pad
			pass
		elif self.padded == DN.WEIGHTPAD:
			#padd weights
			pass




class DN(Enum):
	NONE = -1

	# Both, mPad and pPad
	INPUTPAD = 0
		
	# mPad only
	WEIGHTPAD = 1

	# Stored Plain Weights
	DN.STORED = 2