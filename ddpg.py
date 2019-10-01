import numpy as np
import gym
import critic,actor
import sys
import utils
import buffer_class
import lunar_lander_no_shape
import numpy

class ddpg:
	def __init__(self,params,env,state_size,action_size):
		self.params=params
		self.critic_object=critic.critic(params,env,state_size,action_size)
		self.actor_object=actor.actor(params,env,state_size,action_size,self.critic_object.target_network)
		self.buffer_object=buffer_class.buffer_class(max_length=self.params['max_buffer_size'])

if __name__=='__main__':
	hyper_parameter_name=sys.argv[1]
	params=utils.get_hyper_parameters(hyper_parameter_name)
	params['hyper_parameters_name']=hyper_parameter_name
	env=gym.make(params['env_name'])
	env_test=gym.make("LunarLanderNoShape0.0-v0")
	params['env']=env
	params['seed_number']=int(sys.argv[2])
	try:
		params['show']=sys.argv[3]
		params['show']=True
	except:
		params['show']=False
	utils.set_random_seed(params)
	s0=env.reset()
	agent=ddpg(params,
			   env,
			   state_size=len(s0),
			   action_size=len(env.action_space.low))

	G_li=[]
	for episode in range(params['max_episode']):
		s=env.reset()
		done=False

		while done==False:
			a=agent.actor_object.action_selection(s,train_or_test='train')
			#print(a)
			a=np.clip(a,a_min=env.action_space.low[0],a_max=env.action_space.high[0])
			sp,r,done,_=env.step(np.array(a))
			agent.buffer_object.append(s,a,r,done,sp)
			s=sp
		print("updating Q function ...")
		for _ in range(params['updates_per_episode']):
			agent.critic_object.update(agent.buffer_object,
									   agent.actor_object.target_network)
			agent.actor_object.update(agent.buffer_object,agent.critic_object)



		#now test the learned policy
		s=env_test.reset()
		t=0
		G=0
		done=False
		while done==False:
			a=agent.actor_object.action_selection(s,train_or_test='test')
			#print(a)
			sp,r,done,_=env_test.step(numpy.array(a))
			#agent.buffer_object.append(s,a,r,done,sp)
			s=sp
			t=t+1
			G=G+r
		print("in episode {} we collected return {} in {} timesteps".format(episode,G,t))
		G_li.append(G)
		utils.save(G_li,params)
