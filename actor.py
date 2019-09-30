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

def custom_loss(yTrue,yPred):
	return - K.mean(yPred,axis=0)

class actor:

	def __init__(self,params,env,state_size,action_size,critic):
		self.env=env
		self.params=params
		self.state_size=state_size
		self.action_size=action_size
		self.opt,self.network=self.create_network(critic)
		self.num_updates=0
		_,self.target_network=self.create_network(critic)
		self.target_network.set_weights(self.network.get_weights())
		
	def create_network(self,critic):
		state_input=Input(shape=(self.state_size,))
		#define the network
		h=state_input
		for _ in range(self.params['num_layers']-1):	
			h = Dense(self.params['layer_size'],activation='relu')(h)
		final_a=Dense(self.action_size,activation='tanh')(h)
		final_a=Lambda(lambda x: x*self.env.action_space.high[0])(final_a)
		network=Model(inputs=state_input,outputs=final_a)
		network.compile(loss='mse',optimizer='sgd')

		critic.trainable=False
		q=critic([state_input,final_a])	
		
		if self.params['opt']=='rmsprop':
			opt = optimizers.RMSprop(lr=self.params['learning_rate'])
		elif self.params['opt']=='adam':
			opt = optimizers.Adam(lr=self.params['learning_rate'])

		model = Model(inputs=state_input, outputs=q)
		model.compile(loss=custom_loss,optimizer=opt)
		return model,network

	def update(self,buffer_object,critic):
		if len(buffer_object.storage)<self.params['batch_size']:
			return

		batch=random.sample(buffer_object.storage,self.params['batch_size'])
		s_li=[b['s'] for b in batch]
		s_matrix=numpy.array(s_li).reshape(self.params['batch_size'],self.state_size)
		dummy=numpy.zeros((self.params['batch_size'],1))
		self.opt.fit(x=s_matrix,
					y=dummy,
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
	
	def action_selection(self,s,train_or_test,sigma=0.1):
		a=self.network.predict(numpy.array([s]))
		#print(numpy.max(a))
		if train_or_test=='train':
			noise=numpy.random.normal(loc=0.0, scale=sigma,size=self.action_size)
			noise=noise.reshape(1,self.action_size)
			a=a+noise
		else:
			pass

		a=a[0,:].tolist()
		return a


