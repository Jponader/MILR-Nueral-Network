import math

class activationLayer:

	class relu:
		
		def forwardPass(inputMat):
			print("Relu")

			for i in range(0,len(inputMat)):

				if inputMat[i] < 0:
					inputMat[i]= 0

			return inputMat



	class softmax:
		
		def forwardPass(inputMat):
			print("softmax")
			esum = 0

			for i in range(0,len(inputMat)):
				esum += math.log1p(inputMat[i])

			for i in range(0,len(inputMat)):
				inputMat[i] = math.log1p(inputMat[i])/esum

			return inputMat