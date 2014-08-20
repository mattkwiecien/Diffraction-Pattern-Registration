import numpy as np
from pylab import *
from scipy import *
from PIL import Image
import scipy as sp
import scipy.fftpack as spf
from scipy import misc

def phase_corr(a, b):
	tmp = spf.fft2(a)*spf.fft2(b).conj()
	tmp /= abs(tmp)
	return spf.ifft2(tmp)

def maxval(arr):
	maxi = np.argmax(arr)
	return sp.unravel_index(maxi,arr.shape), arr.max()

def mpix(N_FILES):
	fileset = 'mda{:d}_array.npz.npy'
	max_dict={}
	for i in range(N_FILES):
		fn1=fileset.format(309+(i*2))
		fn2=fileset.format(309+((i+1)*2))
		ob1=np.load(fn1)
		ob2=np.load(fn2)
		description=""
		description=fn1+' and '+fn2

		index, maxx = maxval(phase_corr(ob1,ob2))

		roll_ob2=np.roll( np.roll(ob2,index[0],axis=0) , index[1], axis=1)

		max_dict[i]= description, index, maxx
	return max_dict

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
    
	pixel_dict={}
	pixel_loc=[[256,692],\
	[216,645],[145,588],[190,658],[232,680],[217,640],\
	[222,650],[187,614],[152,622],[232,660],[231,670],\
	[260,690],[185,616],[224,645],[190,623],[171,614],\
	[228,662],[201,633],[179,585],[199,601],[190,590],\
	[260,700],[208,617],[254,650],[198,581],[248,637],\
	[225,632],[250,636],[230,663],[297,727],[237,668],\
	[272,687],[213,628],[254,654],[204,592],[288,694],\
	[271,675],[249,634],[279,675],[208,590],[290,691],\
	[213,580],[257,655],[192,582],[247,634],[213,603],\
	[251,634],[253,640],[278,654],[278,653],[221,601],\
	[260,623],[272,654],[312,684],[324,693],[301,695],\
	[288,688],[325,712],[289,692],[286,659],[241,641]]

	for i in range(61):
		fileset = 'mda{:d}'.format(309+(i*2))

		pixel_dict[i] = fileset, pixel_loc[i]

	for i in range(N_FILES):
		fileloc = '/mnt/xfm0/people/kwiecien/opt_obj/'
		filename = 'mda{:d}_optimized_object.csv'.format(309+(i*2))
		setnum=(309+(i*2))
		output_loc = '/mnt/xfm0/people/kwiecien/opt_obj/final_cropped_images/mda{:d}'.format(309+(i*2))
		print "Fitting fileset {:d}".format(309+(i*2))

		ob = np.genfromtxt(filename, delimiter=',', dtype=complex)

		a=np.where(abs(np.angle(ob[:,1024]))>0)
		b=np.where(abs(np.angle(ob[1024,:]))>0)

		c=[elem for elem in a]
		d=[elem for elem in b]

		yi,yf = c[0][0], c[0][-1]
		xi,xf = d[0][0], d[0][-1]

		new_ob = (ob[(yf-480)+40:yf-40, xi+20:xf-20])
		x_len, y_len = shape(new_ob)

		pixel=pixel_dict[i][1]

		f1_start = pixel[0]
		f2_start = f1_start+440
		
		x = sp.zeros((x_len*y_len))
		y = sp.zeros((x_len*y_len))
		z = sp.zeros((x_len*y_len))
		cnt=0
		for i in range(x_len):
			for j in range(y_len):
				if j>f1_start and j<f2_start:
					pass
				else:
					x[cnt] = i
					y[cnt] = j
					z[cnt] = np.angle(new_ob[i,j])

				cnt+=1

		fitfunc = lambda p, x, y: p[0] + p[1]*y + p[2]*x + p[3]*x*y \
		+ p[4]*(y**2) + p[5]*(y**2)*x + p[6]*(x**2) + p[7]*(x**2)*y \
		+ p[8]*(x**2)*(y**2)

		errfunc = lambda p,x,y,z: abs(fitfunc(p,x,y)-z)

		p0 = [1, 0, 0, 0, 0, 0, 0, 0, 0]


		p1, success = optimize.leastsq(errfunc, p0[:], args=(x,y,z))

		xn = sp.zeros((x_len*y_len))
		yn = sp.zeros((x_len*y_len))
		cnt2=0
		for i in range(x_len):
			for j in range(y_len):
				xn[cnt2] = i
				yn[cnt2] = j

				cnt2 +=1

		count=0
		corr_ob = sp.zeros(shape(new_ob))
		corr = fitfunc(p1,xn,yn)

		for i in range(x_len):
			for j in range(y_len):
				corr_ob[i,j]=corr[count]
				count+=1


		corr_img = np.subtract(np.angle(new_ob), corr_ob)
		corr_img = np.subtract(corr_img, np.average(corr_img))


		outfn='/mnt/xfm0/people/kwiecien/opt_obj/np_arrays/mda{:d}_array.npz'.format(setnum)
		print outfn
		np.save(outfn, corr_img[:,int(f1_start):int(f2_start)])

		#imsave(output_loc+'_full.png', corr_img, cmap=cm.hsv)
		#imsave(output_loc, corr_img[:,int(f1_start):int(f2_start)], cmap=cm.hsv)


