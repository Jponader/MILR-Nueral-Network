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
		partailInput = self.seededRandomTensor((1,self.Tlayer.input_shape[1]))
		checkdata  = gen_math_ops.mat_mul(partailInput, self.Tlayer.kernel)[0][0]
		#CheckPoint, Error
		return self.checkpointed,self.partialData != checkdata

	def forwardPass(self, inputs):
		inputs = math_ops.cast(inputs, self.Tlayer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputs, self.Tlayer.kernel)
		if layer.use_bias:
			outputs = biasLayer.forwardPass(outputs, layer.bias)

		return activationLayer.forwardPass(outputs, layer.activation)

	def kernelSolver(self, inputs, outputs):
		assert self.store is not None, "Nothing Stored"

		m = self.keys['M']
		n = self.keys['N']
		p = self.keys['P']
'''
		if self.padded == DN.CRC:

#Validation To Be Removed
			weight = np.array(self.Tlayer.get_weights())
			weight[0][0][0] = 1
			weight[0][4][4] = 1
			#weight[0][9][4] = 1
			weight[0][3][2] = 1
			weight[0][5][1] = 1
			weight[0][6][6] = 1
			#print(weight[0][:9,:9])
			self.Tlayer.set_weights(weight)


			crcData = self.CRC2D(self.Tlayer.kernel)
			errorLoc = self.CRC2dErrorFinder(self.store, crcData)
			print()
			print(errorLoc)

			errRow = np.unique(errorLoc[:,0])
			errCol, count = np.unique(errorLoc[:,1], return_counts=True)

			print(errRow)
			print(errCol)
			print(count)

			errN = len(errRow)
			errP = len(errCol)

			if errP < errN:
				errP == errN


			if errP * errN > self.keys['M'] * self.keys['P']:
				# reduce the number of errors
					#for err in ranger(0,len(errorLoc)):
				assert errP * errN > self.keys['M'] * self.keys['P'], "To Many Errors for CRC to Recover"

			inputArray = []

			for row in errRow:
				inputArray.append(inputs[:,row])

			inputArray = np.array(inputArray)

			print(inputArray)

			outArray = []

			for col in errCol:
				#Need to modift the outVal so that it takes into account the less computation
				outArray.append(outputs[:, col])
	
			outArray = np.array(outArray)
			print(outArray)
			
			#sol = tf.linalg.solve( inputArray, outArray, adjoint=False, name=None)
			#solvedInput = np.linalg.solve(weights.T, output.T)
			print(inputArray.shape)
			print(outArray.shape)
			sol = np.linalg.lstsq(inputArray, outArray,rcond=-1)
			print(sol)
			

		else:
		'''
		mPad, pPad = self.densePadding(m,n,p)

		if self.padded != DN.NONE:
			inputs = tf.concat([inputs, self.seededRandomTensor((mPad-m,n))],0)
			outputs = tf.concat([outputs,self.store[0]], 0)

		assert np.allclose(self.manIn, inputs,  atol=1e-08), "Input differs after padding"
		assert np.allclose(self.manOut, outputs,  atol=1e-08), "Output differs after padding"

		return tf.linalg.solve( inputs, outputs, adjoint=False, name=None)
		#linalg.lstsq(inputs, outputs, fast=False)
		#return np.linalg.lstsq(inputs, outputs,rcond=-1)[0]
		#return  np.linalg.solve(inputs,outputs)
		#return linalg.solve(inputs, outputs)
		

	def backwardPass(self, outputs):
		if self.checkpointed:
			return self.checkpointData

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
		partailInput = self.seededRandomTensor((1,self.Tlayer.input_shape[1]))
		self.partialData  = gen_math_ops.mat_mul(partailInput, self.Tlayer.kernel)[0][0]
		
# Validatioon - TO BE REMOVED
		self.rewStatus = status
		self.rawIn = inputData
		self.rawKernel = self.Tlayer.kernel
		inputData = math_ops.cast(inputData, self.Tlayer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputData, self.Tlayer.kernel)
		self.rawOut = outputs


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
		crcCost = MP/2

		if crcCost < weightCost:
			minCost = crcCost
			crc = True
		else:
			minCost = weightCost
			crc = False

		if status == STAT.NO_INV:
			self.padded = DN.NONE
			status =  STAT.REQ_INV
			pPad = p
		else:
			if (checkpointCost + minCost) < inputCost:
				self.checkpoint(inputData)
				self.padded = DN.NONE
				pPad = p
			else:
				self.checkpointed = False
				self.padded = DN.INPUTPAD

		if self.padded == DN.NONE:
			if crc:
				self.padded = DN.CRC
				mPad = m
			else:
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

		if self.padded == DN.CRC:
			crcData = self.CRC2D(layer.kernel)

# Validatioon - TO BE REMOVED
		self.manOut = outputs


		if self.padded == DN.WEIGHTPAD:
			self.store = [outputs[m:]]
			outputs = outputs[:m,:p]
		elif self.padded == DN.INPUTPAD:
			self.store = [outputs[m:,:p], outputs[:,p:]]
			outputs = outputs[:m,:p]
		elif self.padded == DN.CRC:
			self.store = crcData
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


# Validatioon - TO BE REMOVED
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

		reKernl = self.kernelSolver(self.rawIn, self.rawOut)
		#if not np.allclose(reKernl, self.Tlayer.kernel, atol=1e-06):
				#print("Error Backward pass")

# END VALIDATION		

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

	# CRC and Possible checkpoint
	CRC = 2

