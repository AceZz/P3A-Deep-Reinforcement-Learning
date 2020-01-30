import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from gym import wrappers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer

class MyModel(tf.keras.Model):
    def __init__(self, input_shape, hidden_units, num_actions): #previously input_shape was num_states
        super(MyModel, self).__init__()
#         self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.flatten = tf.keras.layers.Flatten(input_shape=input_shape, data_format=None)
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.flatten(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output
    
# def create_MyModel(input_shape, hidden_units, num_actions):
#     model = Sequential([
# #         InputLayer(input_shape=input_shape),
#         Flatten(),
#         Dense(256, activation='relu', kernel_initializer='RandomNormal'),
#         Dense(128, activation='relu', kernel_initializer='RandomNormal'),
#         Dense(num_actions, activation='softmax', kernel_initializer='RandomNormal')
#     ])
#     return model


class DQN:
    def __init__(self, input_shape, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(input_shape, hidden_units, num_actions)
#         self.model = create_MyModel(input_shape, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        if len(inputs.shape) == 2: # necessary for batch size of one when playing games
            inputs = np.expand_dims(inputs, axis=0)
        return self.model(np.atleast_3d(inputs.astype('float32')))

    @tf.function
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))


    def get_action(self, states, epsilon):
        """states is actually a single state (past 5 days)"""
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
#             return np.argmax(self.predict(np.atleast_3d(states))[0])
            return np.argmax(self.predict(states)[0])


    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


