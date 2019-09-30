import numpy
from gym.envs.registration import registry, register, make, spec

for k in numpy.arange(0,1.01,0.1):
	k_round=numpy.round(k,decimals=1)
	print(k_round)
	register(
	    id='LunarLanderNoShape'+str(k_round)+'-v0',
	    entry_point='lunar_lander_no_shape.LunarLanderNoShape:NewLunarEnv',
	    max_episode_steps=200,
	    kwargs={
	        'prob_of_shape': k

	    }
	)