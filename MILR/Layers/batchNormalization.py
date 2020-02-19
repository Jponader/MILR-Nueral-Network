from MILR.Layers.layerNode import layerNode


class batchNormalization(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(batchNormalization,self).__init__(layer, prev = prev, next = next)
		config = layer.get_config()
		self.center = config['center']
		self.scale = config['scale']
		self.epsilon = config['epsilon']