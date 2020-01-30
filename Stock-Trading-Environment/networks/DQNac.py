import numpy as np
import tensorflow as tf
import gym
import os
import datetime
import json
import datetime as dt
from gym import wrappers

class MyModel(tf.keras.Model):
    
    tf.keras.backend.set_floatx('float64')
    def __init__(self, input_shape, nbr_filters, kernel_size, strides): #previously input_shape was num_states
        super(MyModel, self).__init__()
        
        M = 1
        N = 6
        L = 6
        
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.conv1 = tf.keras.layers.Conv2D(
            filters = nbr_filters, kernel_size = kernel_size, strides = strides, padding = "valid", 
            kernel_initializer = tf.keras.initializers.TruncatedNormal([M,L,N,32], stddev=0.15)
            )
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters = nbr_filters, kernel_size = kernel_size, strides = strides, padding = "valid", 
            kernel_initializer = tf.keras.initializers.TruncatedNormal([1,1,32,1], stddev=0.15)
            )      
         
            
    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        z = self.conv1(z)
        z = tf.layers.batch_normalization(z)
        z = self.self.conv2(z)
        z = tf.layers.batch_normalization(z)

        
        return z
