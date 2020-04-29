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

	def partialCheckpoint(self):
		partailInput = self.seededRandomTensor((1,*self.Tlayer.input_shape[1:]))
		checkdata  = gen_math_ops.mat_mul(partailInput, self.Tlayer.kernel)[0,:]
		layerError = not np.allclose(checkdata, self.partialData, atol=1e-08)

		print("layer-  ", layerError)

		if self.Tlayer.use_bias:
			checkpointed, self.biasError = biasLayer.partialCheckpoint(self)

			print("bias-   ",self.biasError)

		assert not (layerError == True and self.biasError ==True), "Bias and layer Error"

		return self.checkpointed, layerError or self.biasError

	def forwardPass(self, inputs):
		inputs = math_ops.cast(inputs, self.Tlayer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputs, self.Tlayer.kernel)
		if layer.use_bias:
			outputs = biasLayer.forwardPass(outputs, layer.bias)

		return activationLayer.forwardPass(outputs, layer.activation)

	def kernelSolver(self, inputs, outputs):

		ogWeights = self.Tlayer.get_weights()


		if self.Tlayer.use_bias:
			if self.biasError:
				ogWeights[1] = biasLayer.kernelSolver(self, inputs, outputs)
				self.Tlayer.set_weights(ogWeights)
				self.biasError = False
				return

			outputs = biasLayer.backwardPass(self, outputs)

		m = self.keys['M']
		n = self.keys['N']
		p = self.keys['P']

		mPad, pPad = self.densePadding(m,n,p)

		if self.padded != DN.NONE:
			inputs = tf.concat([inputs, self.seededRandomTensor((mPad-m,n))],0)
			outputs = tf.concat([outputs,self.store[0]], 0)

		#assert np.allclose(self.manIn, inputs,  atol=1e-08), "Input differs after padding"
		#assert np.allclose(self.manOut, outputs,  atol=1e-08), "Output differs after padding"

		#return tf.linalg.solve( inputs, outputs, adjoint=False, name=None)
		#linalg.lstsq(inputs, outputs, fast=False)
		#return np.linalg.lstsq(inputs, outputs,rcond=-1)[0]
		#return  np.linalg.solve(inputs,outputs)
		#return linalg.solve(inputs, outputs)

		ogWeights[0] = tf.linalg.solve( inputs, outputs, adjoint=False, name=None)
		self.Tlayer.set_weights(ogWeights)
		return
		

	def backwardPass(self, outputs):
		if self.checkpointed:
			return self.checkpointData

		if layer.use_bias:
			outputs = biasLayer.backwardPass(self, outputs, data_format = layer.data_format)

		assert self.padded == DN.INPUTPAD, "Non inputPad trying to recover Input"
		assert self.store is not None, "Nothing Stored"

		outputs = tf.concat([outputs,self.store[0]], 0)
		outputs = tf.concat([outputs,self.store[1]],1)
		outT = tf.transpose( outputs, perm=None, conjugate=False, name='transpose')
		kernel = self.padder2D(self.Tlayer.kernel,self.keys['N'], self.keys['pPad'] - self.keys['P'], 1)
		weightsT = tf.transpose( kernel, perm=None, conjugate=False, name='transpose')
		solvedInput = tf.linalg.solve( weightsT, outT, adjoint=False, name=None)
		return tf.transpose( solvedInput, perm=None, conjugate=False, name='transpose')[:self.keys['M'],:self.keys['N']]


	def layerInitilizer(self, inputData, status):

		# partial checkpoint
		partailInput = self.seededRandomTensor((1,*self.Tlayer.input_shape[1:]))
		self.partialData  = gen_math_ops.mat_mul(partailInput, self.Tlayer.kernel)[0,:]
		
# Validatioon - TO BE REMOVED
		self.rewStatus = status
		self.rawIn = inputData
		self.rawKernel = self.Tlayer.kernel
		inputData = math_ops.cast(inputData, self.Tlayer._compute_dtype)
		self.rawOut  = gen_math_ops.mat_mul(inputData, self.Tlayer.kernel)

		layer = self.Tlayer
		assert len(inputData.shape) == 2, "Error: Dense Input Not 2D"
		assert len(layer.kernel.shape) == 2, "Error: Dense Kernel Not 2D"

#Determine Padding Type and Requirments
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

	#Does it need padding
		if self.padded == DN.NONE:
			if n > m:
				self.padded = DN.WEIGHTPAD

# Create Padding
		inputData = self.padder2D(inputData,mPad-m,n,0)
		kernel = self.padder2D(layer.kernel,n, pPad - p, 1)

# Validatioon - TO BE REMOVED
		self.manIn = inputData
		self.manKernel = kernel
		#print("	Padding out - IN - Kern", inputData.shape, kernel.shape)
		#print("	Padding - m - p", mPad, pPad)

		inputData = math_ops.cast(inputData, self.Tlayer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputData, kernel)

# Validatioon - TO BE REMOVED
		self.manOut = outputs

		if self.padded == DN.WEIGHTPAD:
			self.store = [outputs[m:,:p]]
			outputs = outputs[:m,:p]
		elif self.padded == DN.INPUTPAD:
			self.store = [outputs[m:,:p], outputs[:,p:]]
			outputs = outputs[:m,:p]
		else:
			self.store = None

		self.keys ={'M':m, 'N':n, 'P':p, 'mPad':mPad, 'pPad':pPad}
		

#Print Summary Statistics
		print('	Weights: ',self.Tlayer.kernel.shape)
		print('	',self.padded)
		cost = 0
		for i in self.store:
			hold = 1
			for j in i.shape:
				hold = hold * j
			cost += hold
		print('	Cost',cost)
		print('	total Cost', self.cost())


		# Validatioon - TO BE REMOVED
		"""
		assert np.allclose(self.rawOut, outputs,  atol=1e-04), "out wrong"

		if self.padded == DN.WEIGHTPAD or self.padded == DN.INPUTPAD:
			outputCheck = tf.concat([self.rawOut,self.store[0]], 0)
			print(outputCheck.shape)
		if self.padded == DN.INPUTPAD:
			outputCheck = tf.concat([outputCheck,self.store[1]],1)
			inputCheck = tf.concat([self.rawIn, self.seededRandomTensor((mPad-m,n))],0)
			assert np.allclose(inputCheck, self.manIn,  atol=1e-06), "input Reconstruct wrong"

		if self.padded == DN.WEIGHTPAD or self.padded == DN.INPUTPAD:
			assert np.allclose(self.manOut, outputCheck,  atol=1e-04), "out reconstruct wrong"

		
		if self.rewStatus == STAT.REQ_INV:
			reInput = self.backwardPass(outputs)
			if not np.allclose(reInput, self.rawIn, atol=1e-06):
				print("Error Backward pass")

		reKernl = self.kernelSolver(self.rawIn, outputs)
		print(reKernl)
		print(type(reKernl))
		print(self.Tlayer.kernel)
		print(type(self.Tlayer.kernel))
		"""
		# END VALIDATION	

		biasIn = outputs	

		if layer.use_bias:
			outputs, status = biasLayer.layerInitilizer(self, outputs, self.Tlayer.get_weights()[1], status)
			biasLayer.kernelSolver(self, biasIn, outputs)

		return activationLayer.staticInitilizer(outputs, layer.activation, status)


	def cost(self):



		total = 0
		if self.checkpointed:
			cost = 1
			for i in self.checkpointData.shape:
				cost = cost*i
			total = total + cost

		cost = 0
		for i in self.store:
			hold = 1
			if i == None:
				continue
			for j in i.shape:
				hold = hold * j
			cost += hold

		total = total + cost

		return total


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

	# CRC and Possible checkpoint
	CRC = 2

