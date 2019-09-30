import gym,sys,keras
from keras.models import Model,Sequential
from keras.layers import Dense,Input,Concatenate, Lambda,Activation,dot
from keras.initializers import RandomUniform
from keras import optimizers
from keras.constraints import Constraint
from keras import backend as K
import numpy 
import gym
import utils_for_q_learning
import buffer_class
import random
import lunar_lander_no_shape

def func_L2(tensors):
	return -K.sqrt(K.sum(K.square(tensors[0]-tensors[1]),
					axis=1,
					keepdims=True)+ 0.00001)
def func_L1(tensors):
	return -K.sum(K.abs(tensors[0]-tensors[1]),
					axis=1,
					keepdims=True)

class Q_class:

	def __init__(self,params,env,state_size,action_size):
		self.env=env
		self.params=params
		self.state_size=state_size
		self.action_size=action_size
		self.network,self.qRef_li=self.create_network()
		self.num_updates=0
		self.target_network,self.target_qRef_li=self.create_network()
		self.target_network.set_weights(self.network.get_weights())
		self.buffer_object=buffer_class.buffer_class(max_length=self.params['max_buffer_size'])

	def create_network(self):
		state_input=Input(shape=(self.state_size,),name='state_input')

		#define the network
		h=state_input
		for _ in range(self.params['num_layers']):	
			h = Dense(self.params['layer_size'],activation='relu')(h)

		q_output=Dense(self.params['num_points'],name='latent_q')(h)
		action_input=Input(shape=(self.action_size,),name='action_input')
		temp_li=[]
		a_li=[]
		for a_index in range(self.params['num_points']):
			h=state_input

			temp=Dense(self.action_size,
							activation='tanh',
							kernel_initializer=RandomUniform(minval=-.1, maxval=+.1, seed=None),
							bias_initializer=RandomUniform(minval=-1, maxval=+1, seed=None))(h)		
			temp=Lambda(lambda x: x*self.env.action_space.high[0],name="action"+str(a_index))(temp)
			a_li.append(temp)
			layer = Lambda(func_L2)
			temp = layer([temp,action_input])
			temp_li.append(temp)
		merged=Concatenate(axis=-1)(temp_li)

		merged=Lambda(lambda x: x * self.params['temperature'])(merged)
		softmax=Activation('softmax')(merged)
		final_q=dot([q_output,softmax],axes=1, normalize=False)
		model = Model(inputs=[state_input, action_input], outputs=final_q)
		if self.params['opt']=='adam':
			opt = optimizers.Adam(lr=self.params['learning_rate'])
		elif self.params['opt']=='nadam':
			opt = optimizers.Nadam(lr=self.params['learning_rate'])
		elif self.params['opt']=='rmsprop':
			opt = optimizers.RMSprop(lr=self.params['learning_rate'])			
		model.compile(loss='mse',optimizer=opt)


		qRef_li=[]
		for j in range(self.params['num_points']):
			each_qRef=[]
			for i in range(self.params['num_points']):
				layer = Lambda(func_L2)
				each_qRef.append(layer([a_li[i],a_li[j]]))
			each_qRef=Concatenate(axis=-1)(each_qRef)
			each_qRef=Lambda(lambda x: x * self.params['temperature'])(each_qRef)
			each_qRef=Activation('softmax')(each_qRef)
			test_final_q=dot([q_output,each_qRef],axes=1, normalize=False)
			qRef_li.append(test_final_q)
		qRef_li=Model(inputs=state_input,
					  outputs=[Concatenate(axis=1)(a_li),
							   Concatenate(axis=-1)(qRef_li)])

		return model,qRef_li

	def e_greedy_policy(self,s,episode,train_or_test):
		
		if train_or_test=='train':
			#get value of epsilon (depends on episode) and explore
			epsilon=1./numpy.power(episode,1./self.params['policy_parameter'])
			if random.random()<epsilon:
				a=self.env.action_space.sample()
				return a.tolist()

			#if no explore, then exploit
			s_matrix=numpy.array(s).reshape(1,self.state_size)
			aRef_li,qRef_li=self.qRef_li.predict(s_matrix)
			max_index=numpy.argmax(qRef_li)
			aRef_begin,aRef_end=max_index*self.action_size,(max_index+1)*self.action_size
			a=aRef_li[0,aRef_begin:aRef_end]
			'''
			better_a=self.optimizer.optimize_network([numpy.array(s)],
													 a,
													 alpha=params['lr_for_optimization'],
													 max_iterations=params['max_iterations_for_optimization'])
			'''
			return a.tolist()

		elif train_or_test=='test':
			s_matrix=numpy.array(s).reshape(1,self.state_size)
			aRef_li,qRef_li=self.qRef_li.predict(s_matrix)
			aRef_li,qRef_li=self.qRef_li.predict(s_matrix)
			max_index=numpy.argmax(qRef_li)
			aRef_begin,aRef_end=max_index*self.action_size,(max_index+1)*self.action_size
			a=aRef_li[0,aRef_begin:aRef_end]
			'''
			better_a=self.optimizer.optimize_network([numpy.array(s)],
													 a,
													 alpha=0.05,
													 max_iterations=params['max_iterations_for_optimization'])
			'''
			return a


	def update(self):
		#should be changed
		'''
		1-samples a bunch of tuples from the buffer
		2-to compute a*, randomly initializes some a, then does gradien ascent to improve them
		3-gets Q corresponding with best action fro previous step
		4-then performs Q-learning update
		5-from time to time, syncs target network
		'''
		if len(self.buffer_object.storage)<params['batch_size']:
			return
		else:
			pass
		batch=random.sample(self.buffer_object.storage,params['batch_size'])
		s_li=[b['s'] for b in batch]
		sp_li=[b['sp'] for b in batch]
		r_li=[b['r'] for b in batch]
		done_li=[b['done'] for b in batch]
		a_li=[b['a'] for b in batch]
		s_matrix=numpy.array(s_li).reshape(params['batch_size'],self.state_size)
		a_matrix=numpy.array(a_li).reshape(params['batch_size'],self.action_size)
		r_matrix=numpy.array(r_li).reshape(params['batch_size'],1)
		r_matrix=numpy.clip(r_matrix,a_min=-self.params['reward_clip'],a_max=self.params['reward_clip'])
		sp_matrix=numpy.array(sp_li).reshape(params['batch_size'],self.state_size)
		done_matrix=numpy.array(done_li).reshape(params['batch_size'],1)


		next_aRef_li,next_qRef_li=self.target_qRef_li.predict(sp_matrix)
		next_qRef_star_matrix=numpy.max(next_qRef_li,axis=1,keepdims=True)
		label=r_matrix+self.params['gamma']*(1-done_matrix)*next_qRef_star_matrix
		self.network.fit(x=[s_matrix,a_matrix],
						y=label,
						epochs=1,
						batch_size=params['batch_size'],
						verbose=0)
		self.num_updates=self.num_updates+1

		if self.num_updates%params['target_network_period']==0:
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

if __name__=='__main__':
	hyper_parameter_name=sys.argv[1]
	params=utils_for_q_learning.get_hyper_parameters(hyper_parameter_name)
	params['hyper_parameters_name']=hyper_parameter_name
	env=gym.make(params['env_name'])
	params['env']=env
	params['seed_number']=int(sys.argv[2])
	try:
		params['show']=sys.argv[3]
		params['show']=True
	except:
		params['show']=False
	utils_for_q_learning.set_random_seed(params)
	s0=env.reset()
	utils_for_q_learning.action_checker(env)
	Q_object=Q_class(params,env,state_size=len(s0),action_size=len(env.action_space.low))
	G_li=[]
	for episode in range(params['max_episode']):
		

		#train policy with exploration
		s,done=env.reset(),False
		while done==False:
			a=Q_object.e_greedy_policy(s,episode+1,'train')
			sp,r,done,_=env.step(numpy.array(a))
			Q_object.buffer_object.append(s,a,r,done,sp)
			s=sp
		for _ in range(params['updates_per_episode']):
			Q_object.update()

		#test learned policy
		s,t,G,done=env.reset(),0,0,False
		while done==False:
			a=Q_object.e_greedy_policy(s,episode+1,'test')
			sp,r,done,_=env.step(numpy.array(a))
			s,t,G=sp,t+1,G+r
		print("in episode {} we collected return {} in {} timesteps".format(episode,G,t))
		G_li.append(G)
		if episode % 10 == 0 and episode>0:	
			utils_for_q_learning.save(G_li,params)


	utils_for_q_learning.save(G_li,params)