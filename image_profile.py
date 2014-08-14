import numpy as np
from pylab import *
from scipy import *
from PIL import Image

def crop(N_FILES):
	for i in range(N_FILES):
		
		fileloc = '/mnt/xfm0/people/kwiecien/opt_obj/'
		filename = 'mda{:d}_optimized_object.csv'.format(309+(i*2))
		angle_fn = 'mda{:d}_crop_angle'.format(309+(i*2))

		ob = np.genfromtxt(filename, delimiter=',', dtype=complex)

		a=np.where(abs(np.angle(ob[:,1024]))>0)
		b=np.where(abs(np.angle(ob[1024,:]))>0)

		c=[elem for elem in a]
		d=[elem for elem in b]

		yi,yf = c[0][0], c[0][-1]
		xi,xf = d[0][0], d[0][-1]

		new_ob = np.angle(ob[(yi+20):(yf-20), (xi+20):(xf-20)])

		imsave('/mnt/xfm0/people/kwiecien/opt_obj/raw_cropped/'+angle_fn+'.png', new_ob, cmap=cm.hsv)

		return xi, xf, yi, yf

def datafit(N_FILES):
	for i in range(N_FILES):
		fileloc = '/mnt/xfm0/people/kwiecien/opt_obj/raw_cropped/'
		filename = 'mda{:d}_crop_angle'.format(309+(i*2))

		ob = np.genfromtxt(filename, delimiter=',', dtype=complex)

		a=np.where(abs(np.angle(ob[:,1024]))>0)
		b=np.where(abs(np.angle(ob[1024,:]))>0)

		c=[elem for elem in a]
		d=[elem for elem in b]

		yi,yf = c[0][0], c[0][-1]
		xi,xf = d[0][0], d[0][-1]

		new_ob = np.angle(ob[yi+20:yf-20, xi+20:xf-20])

		x_len, y_len = shape(new_ob)
		x_dim = new_ob[x_len-1,:]
		x_diff = np.diff(x_dim[:800])
		x_loc = np.where(abs(x_diff)>1.5)
		x_start = x_loc[0][0]
		x_end = x_loc[0][-1] + 40

		fit_start = new_ob[:, :x_start]
		fit_end = new_ob[:, x_end:]
		fit = np.hstack((fit_start,fit_end))

		#SKETCH
		x_points = fit[1,:]
		y_points = fit[:,1]
		m,n = shape(fit)
		xp=np.linspace(0,m,m, endpoint=False)
		yp=np.linspace(0,n,n, endpoint=False)

		xg,yg = meshgrid(x_points, y_points)

		fitfunc = lambda p,x,y: p[0] + p[1]*y + p[2]*x + p[3]*x*y \
		+ p[4]*(y**2) + p[5]*(y**2)*x + p[6]*(x**2) + p[7]*(x**2)*y \
		+ p[8]*(x**2)*(y**2)
		errfunc = lambda p,x,y: np.subtract(fitfunc(p,x,y).flatten(), fit.flatten())

		p0 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
		p1, success = optimize.leastsq(errfunc, p0[:], args=(xg, yg))

		xt = new_ob[1,:]
		yt = new_ob[:,1]

		xgt,ygt = meshgrid(xt,yt)
		corr = fitfunc(p1,xgt,ygt)
		new_img = np.subtract(new_ob, corr)