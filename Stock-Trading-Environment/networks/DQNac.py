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
    tf.keras.backend.set_floatx('float32')
    #Initial hyperparaters
    
    def __init__(self): 
        
        super(StockActor, self).__init__()
        
        self.tau = 10e-3
        self.learning_rate = 10e-2
        self.gamma = 0.99
        
        self.model = MyModel(32, (1,5), (1,1))       
        self.dense1 = tf.keras.layers.Dense(
                10, activation='softsign', kernel_initializer='GlorotNormal')
        self.dense2 = tf.keras.layers.Dense(
                1, activation='softsign', kernel_initializer='GlorotNormal')        
    
    
    def call(self,inputs):
            
        z = inputs
        z = self.model(z)
        z = tf.squeeze(z)
        z = tf.expand_dims(z, 0)
        z = tf.expand_dims(z, 0)
        z = self.dense1(z)
        z = self.dense2(z)

        return z  
    
        
class StockCritic(tf.keras.Model):
    tf.keras.backend.set_floatx('float32')
    #Initial hyperparaters
    
    def __init__(self): 
        
        super(StockCritic, self).__init__()
        
        self.tau = 10e-3
        self.learning_rate = 10e-4
        self.gamma = 0.99
        #self.action= tf.constant(0.)

        
        self.model = MyModel(32, (1,4), (1,1))
       
        self.dense1 = tf.keras.layers.Dense(
            10, activation='tanh', kernel_initializer='GlorotNormal')
        self.dense2 = tf.keras.layers.Dense(
            10, activation='tanh', kernel_initializer='GlorotNormal')        
        self.dense3 = tf.keras.layers.Dense(
            1, activation='tanh', kernel_initializer='GlorotNormal')
        
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    
        
    def call(self,inputs):                               #inputs=[state,action]
        #print(1)
        z = inputs[0]
        z = self.model(z)
        z = tf.squeeze(z)
        a = [z[0], z[1], inputs[1]]
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
     
    def convert_action(self,action):
    
        if action > 10e-3:
            return np.array([0, action]) #buy stocks with action% of remaining balance
        elif action < -10e-3:
            return np.array([1, -action]) #sell action% of stocks
        else:
            return np.array([2, 0]) #do nothing
    
    
    def convert_action_back(self,action):
        if action[0]==2:
            return tf.squeeze(tf.constant(0,dtype=tf.float32))
        elif action[0]==1:
            return tf.squeeze(tf.constant(-action[1],dtype=tf.float32))
        else:
            return tf.squeeze(tf.constant(action[1],dtype=tf.float32))
    
    def train(self,env,T): 
        
        observations = env.reset()
        observations = np.expand_dims(observations,axis=0)
        observations = np.expand_dims(observations,axis=0)
        
        for t in range(T):
            action = self.act(observations) # observations is actually a single "state" ie past 5 days
            action = tf.squeeze(action)
            action = self.convert_action(action)
            #print(action)
            prev_observations = observations
            observations, reward, done, _ = env.step(action)
            observations = np.expand_dims(observations,axis=0)
            #observations = np.expand_dims(observations,axis=0)
            exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
            self.add_experience(exp)
            
            
            ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
            states = np.asarray([self.experience['s'][i] for i in ids])
            actions = np.asarray([self.convert_action_back(self.experience['a'][i]) for i in ids])
            rewards = np.asarray([tf.dtypes.cast(self.experience['r'][i],tf.float32) for i in ids])
            states_next = np.asarray([self.experience['s2'][i] for i in ids])
            dones = np.asarray([self.experience['done'][i] for i in ids])
            
            #calcul de crit_target(s_next,action_target(s_next))
            action_tar = self.act_tar(states_next)
            print(action_tar)
            action_tar = tf.squeeze(action_tar)
            print(action_tar)
            value_next = self.crit_tar([states_next,action_tar])
            print(value_next)
            value_next = tf.squeeze(value_next)
            print(value_next)
            
            
            Y = rewards + self.gamma* value_next
            
            
            #calcul de crit(s,a)

            Qval = tf.squeeze(self.crit([states_next,actions]))
            
             
            Y = tf.constant(Y, dtype=tf.float32)
            Qval = tf.constant(Qval, dtype=tf.float32)
            
            #update Q
            with tf.GradientTape() as tape:
                loss = tf.math.reduce_sum(tf.square(Y - Qval))
            variables = self.crit.trainable_variables
            gradients = tape.gradient(loss, variables)
            print(loss)
            print(gradients)
            self.optimizer.apply_gradients(zip(gradients, variables))
            print("Done optimize")

        
        
        
        
        
        