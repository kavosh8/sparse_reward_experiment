import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy

def truncate(li):
	#print([len(x) for x in li])
	N=numpy.min([len(x) for x in li])
	return [l[:N] for l in li]
def smooth(li):
	window=10
	y=li
	y_smooth=[numpy.mean(y[max(x-window,0):x+window]) for x in range(len(y))]
	return y_smooth

#[20,31,40,45]
#for hyper_parameter_name in ['10','11','12','13','14','15','16','17','lunar_old']:
#for hyper_parameter_name in [0,1,2,3]:
#for hyper_parameter_name in [20,21,22,23]:
#for hyper_parameter_name in range(40,55):
#for hyper_parameter_name in range(70,82):
#for hyper_parameter_name in range(82,85):
#for hyper_parameter_name in range(85,100):
problems_name=['Pendulum','LunarLander','Bipedal','Ant','Cheetah',
			   'Hopper','InvertedDoublePendulum','InvertedPendulum',
			   'Reacher','Swimmer','Walker2d']
for problem in range(8):
	plt.subplot(4,2,problem+1)
	print(problems_name[problem])
	for setting in [0]:
		hyper_parameter_name=10*problem+setting
		acceptable_len=00
		li=[]
		for seed_num in range(20):
			try:
				temp=numpy.loadtxt("q_learning_results/"+str(hyper_parameter_name)+"/"+str(seed_num)+".txt")
				#print(hyper_parameter_name,numpy.mean(temp[-10:]),len(temp))
				#plt.plot(temp)
				if len(temp)>acceptable_len:
					li.append(temp)
					#plt.plot(temp)
					#print(hyper_parameter_name,seed_num,numpy.mean(temp[-10:]),len(temp))
			except:
				print("problem")
				pass
		#print([len(x) for x in li])
		li=truncate(li)
		print(hyper_parameter_name,
			numpy.mean(li),len(li),
			len(li[0]),
			numpy.mean(numpy.mean(li,axis=0)[-10:]))
		plt.plot(smooth(numpy.mean(li,axis=0)),label=hyper_parameter_name,lw=3)
		#plt.ylim([0,5000])
	plt.title(problems_name[problem])
	plt.legend()
plt.subplots_adjust(wspace=0.5,hspace = 1)
plt.show()