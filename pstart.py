import numpy as np
from scipy import *
from pylab import *

def pstart(ob):

	a=np.where(abs(np.angle(ob[:,1024]))>0)
	b=np.where(abs(np.angle(ob[1024,:]))>0)

	c=[elem for elem in a]
	d=[elem for elem in b]

	yi,yf = c[0][0], c[0][-1]
	xi,xf = d[0][0], d[0][-1]

	new_ob = (ob[(yf-480)+40:yf-40, xi+20:xf-20])
	matshow(np.angle(new_ob))
	show()
	x_start=raw_input('begin?')
	