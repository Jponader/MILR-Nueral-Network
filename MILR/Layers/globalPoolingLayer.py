from MILR.Layers.layerNode import layerNode


class globalPoolingLayer(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(globalPoolingLayer,self).__init__(layer, prev = prev, next = next)
		config = layer.get_config()