import math

from itertools import zip_longest

class loss:

	def mean_sqrd_error(predictions, targets):
		return 0

	def binary_crossentropy(predictions, targets):
		return 0

	def categorical_crossentropy(predictions, targets):
		out = 0
		for p, t in zip_longest(predictions, targets):
				out += t* math.log(p)

		return out
		
	def sparse_categorical_crossentropy(predictions, targets):
		return 1* math.log(predictions[targets])
