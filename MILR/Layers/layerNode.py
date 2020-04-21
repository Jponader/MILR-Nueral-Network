from MILR.status import status as STAT

import sys
import math
import numpy as np
from math import ceil
from zlib import crc32
import tensorflow as tf
from datetime import datetime
from random import randint, seed

CRCCODE = int('9d7f97d6',16)


class layerNode:

	def __init__(self, layer, prev = None, next = None, end = False):
		self.prev = [prev]
		self.next = [next]
		self.end = end
		self.Tlayer = layer
		self.inputLayer = False
		self.checkpointed = False
		self.seed = None

		#These may be remoavable
		self.name = layer.name

	def __str__(self):
		return self.name 
		#+ " Next " + str(len(self.next)) + " Prev " + str(len(self.prev))

	def partialCheckpoint(self):
		#CheckPoint, Error
		return self.checkpointed,False

	def initilize(self, status = STAT.START, inputData = None):
		if status == STAT.START:
			assert self.inputLayer, ("ERROR :Not Start Layer")
			inputData = self.startMetadata()
			status = STAT.NO_INV

		print("\n\n",self,"	", status, self.Tlayer.input_shape, self.Tlayer.output_shape)

		assert inputData is not None, ("ERROR : No input data for next round")

		outputData, status = self.layerInitilizer(inputData, status)
		print('	Checkpointed: ', self.checkpointed)
		if not self.end:
			for n in self.next:
				n.initilize(status, inputData = outputData)
		else:
			#this might vary based on status and layer to be adjusted
			self.outputData = outputData

	# Assumptions
		# Layers do not require checkpointing and status has no changes
		# No additional data is needed for invertibality 
	def layerInitilizer(self, inputData, status):
		return self.Tlayer.call(inputData), status

	def checkpoint(self, inputs):
		self.checkpointed = True
		self.checkpointData = inputs
		cost = 1
		for i in inputs.shape:
			cost = cost*i
		print('	checkpoint: ',inputs.shape, cost)

	def seeder(self):
		if self.seed == None:
			seed()
			self.seed = randint(0,10000)
		return self.seed

	def seededRandomTensor(self, shape):
		np.random.seed(self.seeder())
		return tf.convert_to_tensor(np.random.rand(*shape),  dtype= self.Tlayer.dtype)

	def padder2D(self,inputs, x, y, axis):
		out = self.seededRandomTensor((x,y))
		return tf.concat([inputs,out], axis)

	def CRC2D(self, data):
		shape = data.shape
		output = [np.zeros((shape[0],int(ceil(shape[1]/4)))),np.zeros((int(ceil(shape[0]/4)),shape[1]))]

		for i in range(shape[0]):
			for j in range(int(ceil(shape[1]/4))):
				output[0][i][j] = crc32(data[i,j*4:(j*4)+4],CRCCODE)

		for i in range(int(ceil(shape[0]/4))):
			for j in range(shape[1]):
				output[1][i][j] = crc32(data[i*4:(i*4)+4,j],CRCCODE)	

		return output

	def CRC2dErrorFinder(self, data1, data2):
		results = np.equal(data1[0], data2[0])
		results2 = np.equal(data1[1], data2[1])

		columns = np.argwhere(results == False)
		rows = np.argwhere(results2 == False)

		errorMatrix = []

		for col in columns:
			index = col[1] * 4
			for r in rows:
				if r[1] >= index and r[1]< index +4:
					check = r[0]*4
					if col[0] >= check and col[0]< check +4:
						errorMatrix.append([col[0],r[1]])

		return np.array(errorMatrix, dtype=np.int32)

	def startMetadata(self):
		self.checkpoint = True
		np.random.seed(self.seeder())
		return tf.convert_to_tensor(np.random.rand(1,*self.inputSize[0][1:]),  dtype= self.dtype)

	def setAsInputLayer(self):
		self.inputLayer = True
		self.dtype = self.Tlayer.dtype
		self.inputSize = self.Tlayer.input_shape

	def setNext(self, next):
		if self.next[0] == None:
			self.next[0] = next
		else:
			self.next.append(next)

	def setPrev(self, prev):
		if self.prev[0] == None:
			self.prev[0] = prev
		else:
			self.prev.append(prev)

	def hasNext(self):
		return not (self.end) and self.next[0] != None

	def hasPrev(self):
		return not (self.inputLayer) and self.prev[0] != None

	def isEnd(self):
		self.end = True

	def getNext(self):
		return self.next

	def getPrev(self):
		return self.prev

	def isSplit(self):
		if len(self.next) > 1:
			return True
		else:
			return False

	def isUnion(self):
		if len(self.prev) > 1:
			return True
		else:
			return False