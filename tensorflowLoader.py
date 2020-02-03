from modelController import modelController

from denseLayer import denseLayer
from flattenLayer import flattenLayer
from biasLayer import biasLayer
from activationLayer import activationLayer as act


import h5py
import json



class tensorflowLoader:

	def __init__(self, filepath):
		print("Loading Tensorflow h5py Model")
		self.file = h5py.File(filepath, mode='r')

		config = self.file.attrs.get('model_config')

		print(config)

		if config is None:
			raise ValueError('No model found in config file.')
		config = json.loads(config.decode('utf-8'))

		print(config)


		#https://github.com/tensorflow/tensorflow/blob/cf7fcf164c9846502b21cebb7d3d5ccf6cb626e8/tensorflow/python/keras/saving/hdf5_format.py#L123

