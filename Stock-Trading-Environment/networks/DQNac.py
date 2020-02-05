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
        
        self.norm1=tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters = 1, kernel_size = (1,2), strides = strides, padding = "valid"
            )      
         #kernel_initializer = tf.keras.initializers.TruncatedNormal([1,1,32,1], stddev=0.15)
        self.norm2=tf.keras.layers.BatchNormalization()    
        
            
    
    def call(self,inputs):
        z = self.conv1(inputs)
        z = self.norm1(z)
        z = self.conv2(z)
        z = self.norm2(z)
        return z
    
class StockActor(tf.keras.Model):
    #Initial hyperparaters
    
    def __init__(self): 
        
        super(StockActor, self).__init__()
        
        self.tau = 10e-3
        self.learning_rate = 10e-2
        self.gamma = 0.99
        
        self.model = MyModel(32, (1,5), (1,1))       
        self.dense1 = tf.keras.layers.Dense(
                1, activation='tanh', kernel_initializer='RandomNormal')
        #self.dense2 = tf.keras.layers.Dense(
         #       1, activation='tanh', kernel_initializer='RandomNormal')        
    
    
    def call(self,inputs):
            
        z = inputs
        z = self.model(z)
        z = tf.squeeze(z)
        z = tf.expand_dims(z, 0)
        z = tf.expand_dims(z, 0)
        z = self.dense1(z)

        return z  
    
        
class StockCritic(tf.keras.Model):
    #Initial hyperparaters
    
    def __init__(self): 
        
        super(StockCritic, self).__init__()
        
        self.tau = 10e-3
        self.learning_rate = 10e-4
        self.gamma = 0.99
        self.action= tf.constant(0.)

        
        self.model = MyModel(32, (1,4), (1,1))
       
        self.dense1 = tf.keras.layers.Dense(
            10, activation='tanh', kernel_initializer='RandomNormal')
        self.dense2 = tf.keras.layers.Dense(
            10, activation='tanh', kernel_initializer='RandomNormal')        
        self.dense3 = tf.keras.layers.Dense(
            1, activation='tanh', kernel_initializer='RandomNormal')
        
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    
        
    def call(self,inputs):
        print(1)
        z = inputs
        z = self.model(z)
        z = tf.squeeze(z)
        a = [z[0], z[1], self.action]
        z = tf.convert_to_tensor(a)
        z = tf.expand_dims(z, 0)
        z = self.dense1(z)
        z = self.dense2(z)
        z = self.dense3(z)            

        return z       

class DQN():
    
    def __init__(self, gamma, max_experiences, min_experiences, batch_size, lr):
        
        
        #initialize the buffer
        
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences        
                
        
        
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma

        #initialize nets
        self.act = StockActor()
        self.act_tar = StockActor()
        self.act_tar.set_weights(self.act.get_weights()) 
    
        self.crit = StockCritic()
        self.crit_tar = StockCritic()        
        self.crit_tar.set_weights(self.crit.get_weights())
    
    
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)    
        
        
        
        
        
        
        