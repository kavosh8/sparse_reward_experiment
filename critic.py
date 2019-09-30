from keras.models import Model,Sequential
from keras.layers import Dense,Input,Concatenate, Lambda,Activation,dot
from keras import optimizers
import numpy 
from keras.constraints import Constraint
from keras import backend as K
import gym
import utils
import buffer_class
import random
import sys

class critic:

	def __init__(self,params,env,state_size,action_size):
		self.env=env
		self.params=params
		self.state_size=state_size
		self.action_size=action_size
		self.network=self.create_network()
		self.num_updates=0
		self.target_network=self.create_network()
		self.target_network.set_weights(self.network.get_weights())
		#self.buffer_object=buffer_class.buffer_class(max_length=self.params['max_buffer_size'])
		
	def create_network(self):
		state_input=Input(shape=(self.state_size,))
		action_input=Input(shape=(self.action_size,))
		merged=Concatenate(axis=-1)([state_input,action_input])
		#define the network
		h=merged
		for _ in range(self.params['num_layers']-1):	
			h = Dense(self.params['layer_size'],activation='relu')(h)
		final_q=Dense(1)(h)

		model = Model(inputs=[state_input, action_input], outputs=final_q)
		if self.params['opt']=='rmsprop':
			opt = optimizers.RMSprop(lr=self.params['learning_rate'])
		elif self.params['opt']=='adam':
			opt = optimizers.Adam(lr=self.params['learning_rate'])
		model.compile(loss='mse',optimizer=opt)
		return model

	def update(self,buffer_object,target_actor):
		#should be changed
		'''
		1-samples a bunch of tuples from the buffer
		2-to compute a*, randomly initializes some a, then does gradien ascent to improve them
		3-gets Q corresponding with best action fro previous step
		4-then performs Q-learning update
		5-from time to time, syncs target network
		'''
		if len(buffer_object.storage)<self.params['batch_size']:
			return

		batch=random.sample(buffer_object.storage,self.params['batch_size'])
		s_li=[b['s'] for b in batch]
		sp_li=[b['sp'] for b in batch]
		r_li=[b['r'] for b in batch]
		done_li=[b['done'] for b in batch]
		a_li=[b['a'] for b in batch]
		s_matrix=numpy.array(s_li).reshape(self.params['batch_size'],self.state_size)
		a_matrix=numpy.array(a_li).reshape(self.params['batch_size'],self.action_size)
		r_matrix=numpy.array(r_li).reshape(self.params['batch_size'],1)
		r_matrix=numpy.clip(r_matrix,a_min=-self.params['reward_clip'],a_max=self.params['reward_clip'])
		sp_matrix=numpy.array(sp_li).reshape(self.params['batch_size'],self.state_size)
		done_matrix=numpy.array(done_li).reshape(self.params['batch_size'],1)

		ap_matrix=target_actor.predict(sp_matrix)
		next_q_star_matrix=self.target_network.predict([sp_matrix,ap_matrix])

		label=r_matrix+self.params['gamma']*(1-done_matrix)*next_q_star_matrix

		#print("before",self.target_network.get_weights()[1])
		self.network.fit(x=[s_matrix,a_matrix],
						y=label,
						epochs=1,
						batch_size=self.params['batch_size'],
						verbose=0)
		self.num_updates=self.num_updates+1
		if self.num_updates%self.params['target_network_period']==0:
			self.update_target_net()

	def update_target_net(self):
		network_weights=self.network.get_weights()
		target_weights=self.target_network.get_weights()
		new_target_weights=[]
		for n,t in zip(network_weights,target_weights):
			temp=self.params['target_network_learning_rate']*n+(1-self.params['target_network_learning_rate'])*t
			new_target_weights.append(temp)
		self.target_network.set_weights(new_target_weights)
		#print("update target network")
