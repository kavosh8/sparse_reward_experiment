from keras.models import Model,Sequential
from keras.layers import Dense,Input,Add
from keras import optimizers,initializers
import numpy
import Q_class
import tensorflow as tf
import keras.backend as K


class Q_optimizer_class:
	def __init__(self,Q_object,min_val,max_val):
		self.min_val=min_val
		self.max_val=max_val
		self.Q_object=Q_object
		self.target_network_gradients=K.gradients(self.Q_object.target_network.output,
												  self.Q_object.target_network.input[1])
		self.network_gradients=K.gradients(self.Q_object.network.output,
												  self.Q_object.network.input[1])

	def optimize_target_network(self,state_li,action_li,alpha,max_iterations):
		found_action_li=numpy.copy(action_li)
		s_input=[]
		sess = K.get_session()
		for iteration in range(max_iterations):
			evaluated_gradients = sess.run(self.target_network_gradients[0],
											 feed_dict={self.Q_object.target_network.input[0]: s_input,
											 			self.Q_object.target_network.input[1]: a0_input})
			found_action_li=found_action_li+alpha*evaluated_gradients
			found_action_li=found_action_li.clip(self.min_val,self.max_val)
		return found_action_li

	def optimize_network(self,state_li,action_li,alpha,max_iterations):
		if max_iterations==0:
			return action_li
		found_action_li=numpy.copy(action_li).reshape(1,-1)
		sess = K.get_session()
		for iteration in range(max_iterations):
			evaluated_gradients = sess.run(self.network_gradients[0],
											 feed_dict={self.Q_object.network.input[0]: state_li,
											 			self.Q_object.network.input[1]: found_action_li})
			evaluated_gradients_magnitude=numpy.linalg.norm(evaluated_gradients)
			if evaluated_gradients_magnitude<0.00001:
				break
			evaluated_gradients=evaluated_gradients/evaluated_gradients_magnitude
			found_action_li=found_action_li+alpha*evaluated_gradients
			found_action_li=found_action_li.clip(self.min_val,self.max_val)
		return found_action_li[0]

