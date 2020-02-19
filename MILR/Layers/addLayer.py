from MILR.Layers.layerNode import layerNode
import MILR.status as STAT

class addLayer(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(addLayer,self).__init__(layer, prev = prev, next = next)
		config = layer.get_config()


	def layerInitilizer(self, inputData, status):
		return inputData, status
