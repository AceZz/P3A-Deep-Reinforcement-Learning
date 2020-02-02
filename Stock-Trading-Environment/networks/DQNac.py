import numpy as np
import tensorflow as tf
import gym
import os
import datetime
import json
import datetime as dt
from gym import wrappers

class MyModel(tf.keras.Model):
    
    tf.keras.backend.set_floatx('float32')
    def __init__(self, nbr_filters, kernel_size, strides): #previously input_shape was num_states
        
        super(MyModel, self).__init__()
        
        M = 1
        N = 6
        L = 6
        
        self.conv1 = tf.keras.layers.Conv2D(
            filters = nbr_filters, kernel_size = kernel_size, strides = strides, padding = "valid"
            )
        #,kernel_initializer = tf.keras.initializers.TruncatedNormal([M,L,N,32], stddev=0.15)
        
        self.norm=tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters = 1, kernel_size = (1,2), strides = strides, padding = "valid" 
            )      
         #kernel_initializer = tf.keras.initializers.TruncatedNormal([1,1,32,1], stddev=0.15)
            
    @tf.function
    def call(self, inputs):
        z = inputs
        z = self.conv1(z)
        z = self.norm(z)
        z = self.conv2(z)

        
        return z
    
class StockActor:
    #Initial hyperparaters
    
    def __init__(self): 
        
        #super(StockActor, self).__init__()
        
        self.tau = 10e-3
        self.learning_rate = 10e-2
        self.gamma = 0.99
        
        
        self.model = MyModel(32, (1,4), (1,1))        
        self.dense = tf.keras.layers.Dense(
                1, activation='tanh', kernel_initializer='RandomNormal')
        
    
    @tf.function
    def build_actor(self,inputs):
            z = inputs
            z = self.model(z)
            z = tf.squeeze(z)
            z = tf.expand_dims(z, 0)
            z = self.dense(z)

            return z
        
        
    def predict(self, inputs):
        
        return self.build_actor(inputs.astype('float32'))  
    
        
class StockCritic:
    #Initial hyperparaters
    
    def __init__(self): 
        
        #super(StockActor, self).__init__()
        
        self.tau = 10e-3
        self.learning_rate = 10e-2
        self.gamma = 0.99
        self.model = MyModel(32, (1,4), (1,1))
        self.dense1 = tf.keras.layers.Dense(
            10, activation='tanh', kernel_initializer='RandomNormal')
        self.dense2 = tf.keras.layers.Dense(
            10, activation='tanh', kernel_initializer='RandomNormal')        
        self.dense3 = tf.keras.layers.Dense(
            1, activation='tanh', kernel_initializer='RandomNormal')
    
    
    @tf.function    
    def build_critic(self,inputs,action):
            z = inputs
            z = self.model(z)
            z = tf.squeeze(z)
            a = [z[0], z[1], action[0][0]]
            z = tf.convert_to_tensor(a)
            #z = tf.concat((z,action),axis=0)
            z = tf.expand_dims(z, 0)
            z = self.dense1(z)
            z = self.dense2(z)
            z = self.dense3(z)            

            return z       
        
        
    def predict(self, inputs,action):
        
        return self.build_critic(inputs.astype('float32'),action)      

        
        
        
        
        
        
        
        
        
        