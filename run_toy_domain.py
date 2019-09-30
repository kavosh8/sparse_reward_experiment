import Q_class,Q_optimizer_class
import numpy
import matplotlib as mpl
import utils_for_toy_domain
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

Q_object=Q_class.Q_class(state_size=1,action_size=1)

#define true labels
a_matrix=-20*numpy.random.random(100).reshape(100,1)+10
s_matrix=numpy.zeros_like(a_matrix)
q_matrix=numpy.sin(a_matrix[:,0])*a_matrix[:,0]
#define true labels

for epoch_number in range(500):
	log=Q_object.network.fit(x=[s_matrix,a_matrix],
							y=q_matrix,
							epochs=1)
	#set a threshold to break
	if log.history['loss'][-1]<.5:
		break
	#utils_for_toy_domain.plot_critic(Q_object)


#Q_optimizer_obj=Q_optimizer_class.Q_optimizer_class(Q_object,min_val=-10,max_val=10)
#num_initial_points=100
#a0_input=20*numpy.random.random(num_initial_points).reshape(num_initial_points,1)-10
#a_star=Q_optimizer_obj.optimize(state=[0],a0_input=a0_input)