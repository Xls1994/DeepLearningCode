import numpy as np
def dropout(x,level):
	if level<0 or level>=1:
		raise Exception ('dropout level must be in interval [0,1]')
	retain_prob =1.-level
	sample =np.random.binomial(n=1,p=retain_prob,size=x.shape)
	print sample
	x *=sample
	print x
	x /= retain_prob
	return x
X =np.asarray([1,2,3,4,5,6,7],dtype='float32')
print dropout(X,0.4)
