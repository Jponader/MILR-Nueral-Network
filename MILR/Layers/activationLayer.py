import math

class activationLayer:

	def __init__(self, activationFunc = None):
		print("New Activation Layer")
		if activationFunc == None:
			self.activation = activationLayer.empty
		else:
			self.activation = activationFunc

	def forwardPass(self, inputMat):
		return self.activation.forwardPass(inputMat)

	def compile(self, inputShape):
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