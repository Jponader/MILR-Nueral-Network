from MILR.Layers.layerNode import layerNode


class template(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(template,self).__init__(layer, prev = prev, next = next)
		config = layer.get_config()