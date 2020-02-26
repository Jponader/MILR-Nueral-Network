import MILR.status as STAT

import sys
import math
import numpy as np
import tensorflow as tf
from random import randint, seed
from datetime import datetime



class layerNode:

	def __init__(self, layer, prev = None, next = None, end = False):
		self.prev = [prev]
		self.next = [next]
		self.end = end
		self.Tlayer = layer
		#print(layer.name)
		self.inputLayer = False

		#These may be remoavable
		self.name = layer.name
		#self.inputSize = self.Tlayer.input_shape
		#self.outputSize = self.Tlayer.output_shape


	def __str__(self):
		return self.name 
		#+ " Next " + str(len(self.next)) + " Prev " + str(len(self.prev))

	def initilize(self,status = STAT.START, inputData = None):
		if status == STAT.START:
			assert self.inputLayer, ("ERROR :Not Start Layer")
			inputData = self.startMetadata()
			status = STAT.NO_INV

		print(self)

		assert inputData is not None, ("ERROR : No input data for next round")

		outputData, status = self.layerInitilizer(inputData, status)
		if not self.end:
			for n in self.next:
				n.initilize(status, inputData = outputData)
		else:
			#this might vary based on status and layer to be adjusted
			self.outputData = outputData

	# Assumption of Passthrough Layers will have same input as oputput
	def layerInitilizer(self, inputData, status):
		return inputData, status

	def startMetadata(self):
		self.checkpoint = True
		seed()
		self.seed = randint(0,10000)
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


