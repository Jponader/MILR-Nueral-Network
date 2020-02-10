import numpy as np

from MILR.Layers.layerNode import layerNode



class flattenLayer(layerNode):

	def __init__(self, layer, prev = None):
		super(flattenLayer,self).__init__(prev = prev)
		self.Tlayer = layer
		print(layer.name)
		self.name = layer.name
		#print(layer.get_config())


	def initalize(self, inputSize):
		self.inputSize = inputSize
		return(inputSize[0] ** len(inputSize),1)

	def forwardPass(self, inputMat):
		return inputMat.flatten().T

	def backwardPass(self, inputMat):
		inputMat.T.reshape(self.inputSize)
