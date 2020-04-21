from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

class batchNormalization(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(batchNormalization,self).__init__(layer, prev = prev, next = next)
		#config = layer.get_config()
		#self.center = config['center']
		#self.scale = config['scale']
		#self.epsilon = config['epsilon']

	def layerInitilizer(self, inputData, status):
		#add partial Checkpoint
		out = self.Tlayer.call(inputData)
		return out, STAT.REQ_INV

	#to be configured
	def partialCheckpoint(self):
		#CheckPoint, Error
		return self.checkpointed,False