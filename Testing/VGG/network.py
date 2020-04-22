from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.append('../../')
from MILR.MILR import MILR

import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical, HDF5Matrix

# Helper libraries
import numpy as np
from numpy import ndarray as nd
import array as array
import scipy.io
import time
import pathlib
import random
import h5py

#model = VGG19(weights='imagenet')

#keras.backend.set_learning_phase(0)

# Save Weights
#model.save_weights('weights.h5')
# model.load_weights()

# Save Entire Model
#model.save('model.h5')
model = keras.models.load_model('model.h5')

"""
path_val = '../../../TensorFlow/ImageNet/ILSVRC/Data/CLS-LOC/val'
label_text = 'ILSVRC2012_validation_ground_truth.txt'

data_root = pathlib.Path(path_val)

# use */* for folders containig images
all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]
all_image_paths.sort()


imgs = []
i = 1

for paths in all_image_paths:
	original_image = load_img(paths, target_size=(224, 224)) 
	print(paths)
	numpy_image = img_to_array(original_image)
	input_image = np.expand_dims(numpy_image, axis=0) 
	input_image = preprocess_input(input_image)
	imgs.append(input_image[0])
	if i >= test_length:
		break
	i = i + 1

imgs = np.array(imgs)


#Single Image Classification
img_path = '../ImageNet/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000003.JPEG'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(preds)
print(preds.argmax())
print('Predicted:', decode_predictions(preds, top=3)[0])


# Label Matching From 
# https://calebrob.com/ml/imagenet/ilsvrc2012/2018/10/22/imagenet-benchmarking.html
# https://github.com/calebrob6/imagenet_validation
meta = scipy.io.loadmat("meta.mat")
original_idx_to_synset = {}
synset_to_name = {}

for i in range(1000):
    ilsvrc2012_id = int(meta["synsets"][i,0][0][0][0])
    synset = meta["synsets"][i,0][1][0]
    name = meta["synsets"][i,0][2][0]
    original_idx_to_synset[ilsvrc2012_id] = synset
    synset_to_name[synset] = name

synset_to_keras_idx = {}
keras_idx_to_name = {}
f = open("synset_words.txt","r")
idx = 0
for line in f:
    parts = line.split(" ")
    synset_to_keras_idx[parts[0]] = idx
    keras_idx_to_name[idx] = " ".join(parts[1:])
    idx += 1
f.close()

def convert_original_idx_to_keras_idx(idx):
    return synset_to_keras_idx[original_idx_to_synset[idx]]

f = open(label_text,"r")
labels = f.read().strip().split("\n")
labels = list(map(int, labels))
labels = np.array([convert_original_idx_to_keras_idx(idx) for idx in labels])
labels = to_categorical(labels, 1000)
f.close()

labels = labels.argmax(axis=1)

def top_k_accuracy(y_true, y_pred, k=1):
    argsorted_y = np.argsort(y_pred)[:,-k:]
    return np.any(argsorted_y.T == y_true, axis=0).mean()

global_pred = model.predict(imgs[:test_length], verbose = 1)
global_acc1 = top_k_accuracy(labels[:test_length], global_pred,1)
global_acc5 = top_k_accuracy(labels[:test_length], global_pred,5)

"""
#model.summary()

#print("Type: ")
#print(type(model))

milr = MILR(model)