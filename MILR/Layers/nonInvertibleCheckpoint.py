import tensorflow as tf
from tensorflow import keras
import numpy as np

from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

class nonInvertibleCheckpoint(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(nonInvertibleCheckpoint,self).__init__(layer, prev = prev, next = next)

	# Non Invertible
	# No Weights	
	# Checkpoint when encountered

	def layerInitilizer(self, inputData, status):
		return self.Tlayer.call(inputData), self.metadataHarvester(status)

	def metadataHarvester(self, status):
		if status == STAT.REQ_INV:
			print("			Checkpoint")
		return STAT.NO_INV
