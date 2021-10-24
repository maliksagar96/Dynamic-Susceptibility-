#Calculating Four point Strucure factor and Dynamic susceptibility
#Make sure that the number of particles in all the frames are same or else the code won't work 
#and will have complications. So while linking make sure that goodenough is equal to number of frames.

import time
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import multiprocessing
import concurrent.futures
import warnings


warnings.filterwarnings('ignore')

print("My code")

start = time.time()
mpp = 145 / 512 		 # Microns per pixel, conversion factor
fps = 21 				 # Frame per second
conc = 0.57
k = 0.5
minus_one = -1

j = np.sqrt(np.complex(minus_one))

fname = 'fiveframes.npy'

# loading ndarray having information of x_position, y_position , frame number, particle id respectively
data = np.load(fname)   

frames = int(data[:,2][len(data[:,2])-1] - data[:,2][0] + 1)

data[:, 0] *= mpp                # converting pixel values to micrometers
data[:, 1] *= mpp

b = 1

#Step function which just spits out 0 or 1 depending on whether the input particle has moved greater then some distance or not
#Overlap Function
def overlapFn(b, x_t1,x_t2,y_t1,y_t2):
	overlap = 0.3 * b - (((x_t2 - x_t1)**2+(y_t2 - y_t1)**2)**0.5)
	heaviside_overlap=np.where(overlap<0,0,1)
	return heaviside_overlap

#Fourier transform of overlap function
#q = Fourier variable
def FT(q,heaviside_overlap,x_t1,x_t2,y_t1,y_t2):
	j = np.sqrt(np.complex(minus_one))
	return 	np.sum(heaviside_overlap*np.exp(-j*(q*x_t1+q*y_t1)))

#Minimum window is 1 and not zero.
def calcZeta(i, window, s4, chi4, zeta4r, zeta4Im):
	
	s = 0
	chi = 0
	zeta = 0

	for ti in range(i, i+window):

		Ns = 0 						#Number of slow particles
		Ns2 = 0 					#Ns square
		Wqt = 0 					#Fourier transform of overlap function  
		W_qt = 0          			#It is with minus q 
		loopCounter = 0				#For time averaging

		for tj in range(0, frames):
			if(ti + tj < frames):
				ovrlap = overlapFn(b,x[:,tj],x[:,ti+tj],y[:,tj],y[:,ti+tj])
				ovrlapsum = sum(ovrlap) 
				Ns = Ns + ovrlapsum
				Ns2 = Ns2 + ovrlapsum * ovrlapsum
				ft = FT(k, ovrlap, x[:,tj],x[:, ti+tj], y[:,tj], y[:, ti+tj])
				ft_ = FT(-k, ovrlap, x[:,tj],x[:, ti+tj], y[:,tj], y[:, ti+tj])
				Wqt = Wqt + ft
				W_qt = W_qt  + ft * ft_
				loopCounter = loopCounter + 1
		
		s = (((W_qt)/loopCounter) - (Wqt/loopCounter)**2)/N
		chi = ((Ns2/loopCounter) - (Ns/loopCounter)**2)/N
		zeta = ((chi/s -1)/(k**2))
		s4[ti] = s
		chi4[ti] = chi
		zeta4r[ti] = np.real(zeta)
		zeta4Im[ti] = np.imag(zeta)
		
if __name__ == "__main__":
	        
	N = int(np.shape(np.where(data[:, 2] == data[:,2][0]))[1])		#Total number of particles in frame 0
	print("Number of particles = ", N)
	
	x = np.zeros((N, frames), dtype = float)
	y = np.zeros((N, frames), dtype = float)

	zeta4 = np.zeros(frames, dtype = complex)

	for t in range(int(data[:,2][0]), frames):
		frameIndex = np.where(data[:,2] == t)         # Index of the ith(or tth if you like) frame 
		datafrm = data[frameIndex]                        # Data of that frame
		datafrm = datafrm[datafrm[:, 3].argsort()]	
		x[:, t] = datafrm[:, 0]
		y[:, t] = datafrm[:, 1]


	s4 = multiprocessing.Array('d', frames)
	chi4 = multiprocessing.Array('d', frames)
	zeta4r = multiprocessing.Array('d', frames)
	zeta4Im = multiprocessing.Array('d', frames)

	processes = []

	for i in range(frames):	
		p = multiprocessing.Process(target = calcZeta, args = (i, 1, s4, chi4, zeta4r, zeta4Im))
		p.start()
		processes.append(p)
		
	for process in processes:
		process.join() 
		
	for i in range(frames):
		zeta4[i] = zeta4r[i] + j * zeta4Im[i] 

	
	np.save('zeta4_0.76',zeta4)	

