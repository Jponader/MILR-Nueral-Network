import tensorflow as tf
from tensorflow import keras
import numpy as np

class MILR:

	def __init__(self, model):
		self.model = model
		print("I have a model to work with")
		print(model)

		print(model.layers)