import numpy as np
import tensorflow as tf
import gym
import os
import datetime
import json
import datetime as dt
from gym import wrappers

class MyModel(tf.keras.Model):
    
    def __init__(self, input_shape, nbr_filters, kernel_size, strides,num_actions): #previously input_shape was num_states
        super(MyModel, self).__init__()
        
        M = input_shape[0]
        N = input_shape[1]
        L = input_shape[2]
        
        self.conv1 = tf.keras.layers.Conv2D(
            filters = nbr_filters, kernel_size = kernel_size, stride = strides, padding = "valid", 
            kernel_initializer = tf.keras.initializers.TruncatedNormal([M,L,N,32], stddev=0.15)
            )
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters = nbr_filters, kernel_size = kernel_size, stride = strides, padding = "valid", 
            kernel_initializer = tf.keras.initializers.TruncatedNormal([1,1,32,1], stddev=0.15)
            )      
         
            
    @tf.function
    def call(self, inputs):
        z = self.self.conv1(inputs)
        z = tf.layers.batch_normalization(z)
        z = self.self.conv2(z)
        z = tf.layers.batch_normalization(z)

        
        return z
