from keras.models import Sequential#to create a models with inputs and outputs 
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2#old library of openCV
import os#comes with standard library of the python
import numpy as np
import sys

from numpy import genfromtxt#to create arrays for the tabular data
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *

np.set_printoptions(threshold=sys.maxsize)

#loading the pretrained model
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.2):
    #this is the loss function
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=None)
    # Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=None)
    # subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

#here we have  create the triplet loss as the loss function 
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

database = {}
database["Kavindu"] = img_to_encoding("images/kavindu.jpg", FRmodel)
database["Deshan"] = img_to_encoding("images/deshan.jpg", FRmodel)
database["Suneetha"] = img_to_encoding("images/suneetha.jpg", FRmodel)
database["Anura"] = img_to_encoding("images/anura.jpg", FRmodel)
database["Ravin"] = img_to_encoding("images/ravin.jpg", FRmodel)

def who_is_it(image, database, model):
    
    # Compute the target "encoding" for the image. Use img_to_encoding() see example above.
    encoding = img_to_encoding_no_path(image, model)
    # Initialize "min_dist" to a large value, say 100 
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():#db_enc is database encoding
        # Compute L2 distance between the target "encoding" and the current "emb" from the database
        dist = np.linalg.norm(db_enc - encoding)
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist<min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.7:
        print("No one :(")
        decision = "No one :("
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        decision = identity+" :)"
        
    return min_dist, identity, decision
  
def capture(image):
    min_dist, identity, decision = who_is_it(image, database, FRmodel) 
    
    return decision
