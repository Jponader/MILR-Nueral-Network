from MILR.Layers.layerNode import layerNode
import MILR.status as STAT

class batchNormalization(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(batchNormalization,self).__init__(layer, prev = prev, next = next)
		config = layer.get_config()
		self.center = config['center']
		self.scale = config['scale']
		self.epsilon = config['epsilon']

	def layerInitilizer(self, inputData, status):
		return inputData, status