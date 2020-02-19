import numpy as np

from MILR.Layers.layerNode import layerNode
import MILR.status as STAT

from tensorflow.keras import layers as L
import tensorflow as tf



class flattenLayer(layerNode):

	def __init__(self, layer, prev = None, next = None):
		super(flattenLayer,self).__init__(layer, prev = prev, next = next)

	def layerInitilizer(self, inputData, status):
		return inputData, status

	def initalize(self, inputSize):
		self.inputSize = inputSize
		return(inputSize[0] ** len(inputSize),1)

	def forwardPass(self, inputMat):
		return inputMat.flatten().T

	def backwardPass(self, inputMat):
		inputMat.T.reshape(self.inputSize)
