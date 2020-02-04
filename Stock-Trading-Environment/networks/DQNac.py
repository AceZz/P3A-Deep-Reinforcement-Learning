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
        
        self.enter=tf.keras.layers.InputLayer(input_shape= (1,6,6))
        self.conv1 = tf.keras.layers.Conv2D(
            filters = nbr_filters, kernel_size = kernel_size, strides = strides, padding = "valid"
            )
        #,kernel_initializer = tf.keras.initializers.TruncatedNormal([M,L,N,32], stddev=0.15)
        
        self.norm=tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters = 1, kernel_size = (1,2), strides = strides, padding = "valid",name=out 
            )      
         #kernel_initializer = tf.keras.initializers.TruncatedNormal([1,1,32,1], stddev=0.15)
            
        self.model = tf.keras.Sequential()
            
    #@tf.function
    def build_model(self):
        self.model.add(self.enter)
        self.model.add(self.conv1)
        self.model.add(self.norm)
        self.model.add(self.conv2)
        return self.model
    
class StockActor:
    #Initial hyperparaters
    
    def __init__(self): 
        
        #super(StockActor, self).__init__()
        
        self.tau = 10e-3
        self.learning_rate = 10e-2
        self.gamma = 0.99
        
        mod = MyModel(32, (1,5), (1,1))
        self.model = mod.build_model()        
        self.dense1 = tf.keras.layers.Dense(
                1, activation='tanh', kernel_initializer='RandomNormal')
        #self.dense2 = tf.keras.layers.Dense(
         #       1, activation='tanh', kernel_initializer='RandomNormal')        
    
    #@tf.function
    def build_actor(self):
        
        self.model.add(self.dense1)
        #self.model.add(self.dense2)    
            #z = inputs
            #z = self.model(z)
            #z = tf.squeeze(z)
            #z = tf.expand_dims(z, 0)
            #z = self.dense(z)

        return self.model
        
        
    def predict(self,inputs):
        
        return self.model.predict(inputs.astype('float32'))  
    
        
class StockCritic:
    #Initial hyperparaters
    
    def __init__(self): 
        
        #super(StockActor, self).__init__()
        
        self.tau = 10e-3
        self.learning_rate = 10e-4
        self.gamma = 0.99
        self.action= tf.constant([[[[0.0]]]])

        mod = MyModel(32, (1,4), (1,1))
        self.model = mod.build_model()
        self.model.get_layer()
        #self.addi = tf.keras.layers.add([self.model,self.action])
        #self.conca = tf.keras.layers.Concatenate([self.model,self.action])
        self.dense1 = tf.keras.layers.Dense(
            10, activation='tanh', kernel_initializer='RandomNormal')
        self.dense2 = tf.keras.layers.Dense(
            10, activation='tanh', kernel_initializer='RandomNormal')        
        self.dense3 = tf.keras.layers.Dense(
            1, activation='tanh', kernel_initializer='RandomNormal')
        
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        
        
    
    
    #@tf.function    
    def build_critic(self):
        #self.model.add(self.conca)
        self.model.add(self.dense1)
        self.model.add(self.dense2)
        self.model.add(self.dense3)
            
            #z = inputs
            #z = self.model(z)
            #z = tf.squeeze(z)
            #a = [z[0], z[1], action[0][0]]
            #z = tf.convert_to_tensor(a)
            #z = tf.expand_dims(z, 0)
            #z = self.dense1(z)
            #z = self.dense2(z)
            #z = self.dense3(z)            

        return self.model       
        
        
    def predict(self, inputs,action):
        
        self.action = action
        
        return self.model.predict(inputs.astype('float32'))
    
    def optimize(self):
        return self.model.summary()#self.model.trainable_variables

        
            
        
        
        
        
        
        
        
        