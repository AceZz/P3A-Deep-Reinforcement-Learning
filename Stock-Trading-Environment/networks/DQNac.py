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
        
        self.norm_layer = tf.keras.layers.LayerNormalization()

        self.conv1 = tf.keras.layers.Conv2D(
            filters = nbr_filters, kernel_size = kernel_size, strides = strides, padding = "valid"
            )
        
        
        self.norm1=tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters = 1, kernel_size = (1,2), strides = strides, padding = "valid"
            )      
   
        self.norm2=tf.keras.layers.BatchNormalization()    
        
            
    
    def call(self,inputs):
        z = self.norm_layer(inputs)
        z = self.conv1(z)
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
        
        self.model = MyModel(32, (1,3), (1,1))       
        self.dense1 = tf.keras.layers.Dense(
                10, activation='tanh', kernel_initializer='GlorotNormal')
        self.dense2 = tf.keras.layers.Dense(
                1, activation='tanh', kernel_initializer='GlorotNormal')        
    
    
    def call(self,inputs):
            
        z = inputs
        z = self.model(z)
        z = tf.squeeze(z)
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

        
        self.model = MyModel(32, (1,3), (1,1))
        
        self.norm_layer = tf.keras.layers.LayerNormalization()
        self.dense_input_action = tf.keras.layers.Dense(
            2, activation='tanh', kernel_initializer='glorot_uniform')
        
        self.dense1 = tf.keras.layers.Dense(
            10, activation='tanh', kernel_initializer='glorot_uniform')
        self.dense2 = tf.keras.layers.Dense(
            10, activation='tanh', kernel_initializer='glorot_uniform')        
        self.dense3 = tf.keras.layers.Dense(
            1, activation='tanh', kernel_initializer='glorot_uniform')
        
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    
        
    def call(self,inputs):                                    #inputs=[[Bacth,state],[Bacth,action]]
        #print(1)
        z = inputs[0]
        z = self.model(z)
        z = tf.squeeze(z)                                     #z = [Batch, 1,2]
        inpunts_action = self.norm_layer(inputs[1])
        inpunts_action = self.dense_input_action(inpunts_action)   #inpunts_action = [1,Batch, 2]
        inpunts_action = tf.squeeze(inpunts_action)           #inpunts_action = [Batch, 2]
        z = tf.concat([z,inpunts_action], 1)                  #z = [Batch,4]
        #z = tf.expand_dims(z, 0)
        z = self.dense1(z)
        z = self.dense2(z)
        z = self.dense3(z)            

        return z       

class DQN():
    
    def __init__(self, gamma, max_experiences, min_experiences, batch_size, lr_crit,lr_act):
        
        
        #initialize the buffer
        
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences        
                
        
        
        self.batch_size = batch_size
        self.optimizer_crit = tf.optimizers.Adam(lr_crit)
        self.optimizer_act = tf.optimizers.Adam(lr_act)
        self.gamma = gamma
        
        
        #initialize nets
        self.act = StockActor()
        self.act_tar = StockActor()
        self.act_tar.set_weights(self.act.get_weights()) 
    
        self.crit = StockCritic()
        self.crit_tar = StockCritic()        
        self.crit_tar.set_weights(self.crit.get_weights())
        
        self.tau = tf.constant(1e-2,dtype=tf.float32)
        
        
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
            return tf.constant(0,dtype=tf.float32)
        elif action[0]==1:
            return tf.constant(-action[1],dtype=tf.float32)
        else:
            return tf.constant(action[1],dtype=tf.float32)
    
    def train(self,env,T,table_loss,table_actions_explored,alpha,sigma): 
        
        observations = env.reset()
        observations = np.expand_dims(observations,axis=0)
        observations = np.expand_dims(observations,axis=0)
        observations = tf.convert_to_tensor(observations)   #[1,1,6,6]
        
        for t in range(T):

            #print(observations)
            action = self.act(observations) # observations is actually a single "state" ie past 5 days
            action = tf.squeeze(action)+alpha*np.random.normal(0,sigma)        #np.random.normal(0,0.05)
            table_actions_explored.append(action.numpy())
            action = self.convert_action(action)                        #[1,1]
            #print(action)
            prev_observations = tf.convert_to_tensor(observations)
            prev_observations = tf.squeeze(prev_observations, axis=0)   #[1,6,6]
            #print(prev_observations )
            observations, reward, done, _ = env.step(action)
            observations = np.expand_dims(observations,axis=0)          #[1,6,6]
   
            exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
            self.add_experience(exp)
            observations = np.expand_dims(observations,axis=0)
            observations = tf.convert_to_tensor(observations)           #[1,1,6,6]
            
            ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
            states = np.asarray([self.experience['s'][i] for i in ids])                                # [Batch,1,6,6]
            actions = np.asarray([[self.convert_action_back(self.experience['a'][i])] for i in ids])   # [Batch,1]
            rewards = np.asarray([tf.dtypes.cast(self.experience['r'][i],tf.float32) for i in ids])    # [Batch,1]
            states_next = np.asarray([self.experience['s2'][i] for i in ids])                          # [Batch,1,6,6]
            dones = np.asarray([self.experience['done'][i] for i in ids])
            
            #calcul de crit_target(s_next,action_target(s_next))

            action_tar = self.act_tar(states_next)
            #action_tar = tf.squeeze(action_tar)
            
            #creation of a batch of input state_action
            states_act_next = [tf.convert_to_tensor(states_next),action_tar] 
            #print(states_act)
            
            value_next = self.crit_tar(states_act_next)
            value_next = tf.squeeze(value_next)
            
            
            Y = rewards + self.gamma* value_next               # Y=[Batch]
            Y = tf.constant(Y, dtype=tf.float32)
            
            
            #update Q
            states = tf.convert_to_tensor(states)
            actions = tf.convert_to_tensor(actions)
            states_act = [states,actions]

            with tf.GradientTape(persistent=True) as tape:
                #calcul de crit(s,a)
                tape.watch(actions)
                tape.watch(states)
                Qval = tf.squeeze(self.crit([states,actions]))
                Qval = tf.constant(Qval, dtype=tf.float32)      # Qval=[Batch]
                loss = tf.math.reduce_sum(tf.square(Y - Qval))
                
                act_calc = self.act(states)
                act_calc = tf.squeeze(act_calc, axis=0)
                
            table_loss.append(loss)    
            crit_variables = self.crit.trainable_variables
            act_variables = self.act.trainable_variables    
            
            crit_gradients = tape.gradient(loss, crit_variables)

            self.optimizer_crit.apply_gradients(zip(crit_gradients, crit_variables))
            #print("Done optimize crit")
            

            crit_gradients_wrt_actions = tape.gradient(Qval, actions)
            act_gradients = tape.gradient(act_calc,act_variables,-crit_gradients_wrt_actions)
            self.optimizer_act.apply_gradients(zip(act_gradients, act_variables))
            #print("Done optimize act")
            
            del tape
            
            w_act = np.add(np.multiply(self.tau,self.act.get_weights()),np.multiply(tf.constant(1,dtype=tf.float32)-self.tau,self.act_tar.get_weights()))
            self.act_tar.set_weights(w_act)

            w_crit = np.add(np.multiply(self.tau,self.crit.get_weights()),np.multiply(tf.constant(1,dtype=tf.float32)-self.tau,self.crit_tar.get_weights()))
            self.crit_tar.set_weights(w_crit)            
            
        return self.act
        
        
        
        