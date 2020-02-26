from MILR.Layers.layerNode import layerNode
import MILR.status as STAT

class addLayer(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(addLayer,self).__init__(layer, prev = prev, next = next)
		self.inputData = []

	def initilize(self,status = STAT.START, inputData = None):
		assert status != STAT.START, "ERROR: Add Layer Cannpt be a Start Layer"
		self.inputData.append(inputData)

		if self.canStartInilize():
			print(self)

			assert inputData is not None, ("ERROR : No input data for next round")

			outputData, status = self.layerInitilizer(self.inputData, status)
			if not self.end:
				for n in self.next:
					n.initilize(status, inputData = outputData)
			else:
				#this might vary based on status and layer to be adjusted
				self.outputData = outputData

	def canStartInilize(self):
		if len(self.inputData) == len(self.prev):
			return True
		else:
			return False


	def layerInitilizer(self, inputData, status):
		out = self.Tlayer.call(inputData)
		return out, status


