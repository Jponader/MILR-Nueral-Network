import math

from MILR.Layers.layerNode import layerNode


class activationLayer(layerNode):

	def __init__(self, layer, prev = None, next = None):
		super(activationLayer,self).__init__(layer, prev = prev, next = next)
		self.func = layer.get_config()['activation']

	def layerInitilizer(self, inputSize):
		print("activation")
		return inputSize

# if sublayer use the functions, but not have own object


	def forwardPass(self, inputMat):
		return self.activation.forwardPass(inputMat)

	def initalize(self, inputShape):
		return	inputShape

	class empty:

		def forwardPass(inputMat):
			return inputMat


	class relu:
		
		def forwardPass(inputMat):
			for i in range(0,len(inputMat)):

				if inputMat[i] < 0:
					inputMat[i]= 0

			return inputMat



	class softmax:
		
		def forwardPass(inputMat):
			esum = 0

			for i in range(0,len(inputMat)):
				esum += math.log1p(inputMat[i])

			for i in range(0,len(inputMat)):
				inputMat[i] = math.log1p(inputMat[i])/esum

			return inputMat

	class sigmoid:

		def forwardPass(inputMat):
			for i in range(0,len(inputMat)):
				inputMat[i] = 1 / (1 + math.exp(-inputMat[i]))

				return inputMat