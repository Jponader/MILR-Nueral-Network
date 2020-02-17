import math
import numpy as np
from random import seed
from random import randint
from random import random
from datetime import datetime
from itertools import zip_longest

from MILR.Layers.layerNode import layerNode



class convolutionLayer(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(convolutionLayer,self).__init__(layer, prev = prev, next = next)
		print(layer.get_config())
		config = layer.get_config()
		self.stride = config['strides']
		self.filters = config['filters']
		self.kernel = config['kernel_size']
		self.hasBias = config['use_bias']


		

	
		