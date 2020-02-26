from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops


from MILR.Layers.activationLayer import activationLayer
from MILR.Layers.biasLayer import biasLayer
from MILR.Layers.layerNode import layerNode
import MILR.status as STAT


import math
import numpy as np
from random import seed
from random import randint
from random import random
from datetime import datetime
from itertools import zip_longest


class denseLayer(layerNode):

	def __init__(self, layer, prev = None, next = None):
		super(denseLayer,self).__init__(layer, prev = prev, next = next)
		#config = layer.get_config()
		#self.units = config['units']
		#Common
		#self.hasBias = config['use_bias']
		#self.activationFunc = config['activation']


	def layerInitilizer(self, inputData, status):
		out = self.Tlayer.call(inputData)
		out2 = self.forwardPass(inputData)

		assert np.allclose(out, out2,  atol=1e-10), 'ERROR Different Dense Functions'
		return out2, status


	def forwardPass(self, inputs):
		# This function is based off of Keras.Dense.Call()
		layer = self.Tlayer
		inputs = math_ops.cast(inputs, layer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputs, layer.kernel)

		if layer.use_bias:
			outputs = biasLayer.forwardPass(outputs, layer.bias)

		return activationLayer.forwardPass(outputs, layer.activation)
