import numpy
from scipy.special import softmax as sft

def Q(a,anchor_li,q_li,temperature):
	aDiff=[-temperature*numpy.linalg.norm(numpy.array(a)-numpy.array(z)) for z in anchor_li]
	weights=sft(aDiff)
	out=numpy.dot(weights,numpy.array(q_li))
	return out

numpy.random.seed(111213)
for problem in range(1000000000):
	print("problem # {}".format(problem))
	action_dimension=numpy.random.randint(1,5)
	num_anchor_points=numpy.random.randint(2,10)

	anchor_li=[]
	q_li=[]
	for _ in range(num_anchor_points):
		anchor_point=numpy.random.uniform(low=-1, high=1, size=action_dimension)
		anchor_li.append(anchor_point)
		q_of_anchor_point=numpy.random.uniform(low=-1, high=1)
		q_li.append(q_of_anchor_point)
		temperature=numpy.random.uniform(low=0, high=1)
	anchor_q_li=[Q(a,anchor_li,q_li,temperature) for a in anchor_li]
	print("*********")
	print("*********")
	print("*********")
	print("*********")
	print("anchor li",anchor_li)
	print("q li",q_li)
	print("temperature",temperature)
	print("anchor q li",anchor_q_li)
	max_of_anchor_q_li=numpy.max(anchor_q_li)
	print("max_of_anchor_q_li",max_of_anchor_q_li)

	random_action_li=[]
	for _ in range(10000):
		random_action=numpy.random.uniform(low=-5, high=5, size=action_dimension)
		random_action_li.append(random_action)
	q_of_random_action_li=[Q(a,anchor_li,q_li,temperature) for a in random_action_li]
	max_q_of_random_action_li=numpy.max(q_of_random_action_li)
	arg_max_q_of_random_action_li=numpy.argmax(q_of_random_action_li)
	arg_max_q_of_random_action_li=random_action_li[arg_max_q_of_random_action_li]
	if len(arg_max_q_of_random_action_li)==0:
		assert False
	print("max_q_of_random_action_li",max_q_of_random_action_li)
	print("arg_max_q_of_random_action_li",arg_max_q_of_random_action_li)
	if max_q_of_random_action_li>max_of_anchor_q_li:
		print("found a counter example ...")
	print("*********")
	print("*********")
	print("*********")
	print("*********")


assert False