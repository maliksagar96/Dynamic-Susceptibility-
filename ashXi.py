#Calculating Four point Strucure factor and Dynamic susceptibility
#written by Ashish Joy


#Calculating Four point Strucure factor and Dynamic susceptibility
#written by Ashish Joy

import time
start=time.time()
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import warnings
warnings.filterwarnings('ignore')

mpp=145/512  #microns per pixel, conversion factor
fps=21 # Frame per second

frame_no=5
conc=0.75
k=0.5

minus_one=-1

j=np.sqrt(np.complex(minus_one))

a=np.load('fiveframes.npy')   # loading ndarray having information of x_position, y_position , frame number, particle id respectively

##Sorted out.
b=1
def overlap_fn(b,pos_x_t1,pos_x_t2,pos_y_t1,pos_y_t2):
	overlap=0.3*b-(((pos_x_t2-pos_x_t1)**2+(pos_y_t2-pos_y_t1)**2)**0.5)
	heaviside_overlap=np.where(overlap<0,0,1)
	return heaviside_overlap

#Fourier transform of overlap function
def W(q,heaviside_overlap,pos_x_t1,pos_x_t2,pos_y_t1,pos_y_t2):
	j=np.sqrt(np.complex(minus_one))
	return 	np.sum(heaviside_overlap*np.exp(-j*(q*pos_x_t1+q*pos_y_t1)))

blue=np.where(a[:,2]==1)    
air=a[blue]
N=len(air[:,0])

print("Number of particles = ", N)

pos_x=np.zeros((len(air[:,0]),frame_no),dtype=float)
pos_y=np.zeros((len(air[:,0]),frame_no),dtype=float)

for t in range(0,frame_no):
	a_ind=np.where(a[:,2]==t)
	a_frm=a[a_ind]
	a_frm[:,0]*=mpp
	a_frm[:,1]*=mpp
	particle_unsort=a_frm[:,3]
	index__=particle_unsort.argsort()
	a_frm=a_frm[index__]
	pos_x[:,t]=a_frm[:,0]
	pos_y[:,t]=a_frm[:,1]	

S4_q_t=[]
chi4_t=[]
eta_t=[]

# foo = 0
# foo1 = 3
# heav=overlap_fn(b,pos_x[:,foo],pos_x[:,foo1],pos_y[:,foo],pos_y[:,foo1])
# print(heav)

# counter = 0
# for i in range(np.shape(heav)[0]):
# 	if(heav[i] == 0):
# 		counter+=1

# print(counter)

for ti in range(0,frame_no):
	#coutner variable to take averages
    loop_n = 0        
    W_init = 0
    W_init_= 0
    N_s_init = 0
    N_s_init_ = 0
    # print("I = ",ti)
    for t_ in range(0, frame_no):
        if ti+t_<frame_no:
            # print("J = ", t_+ti)
            heav=overlap_fn(b,pos_x[:,t_],pos_x[:,t_+ti],pos_y[:,t_],pos_y[:,t_+ti])
            Ns=heav
            N_s=sum(Ns)           
            N_s_init=N_s*N_s+N_s_init                   #Sigma(Nsi^2)
            N_s_init_=N_s+N_s_init_                       #Sigma(Nsi)
            W_sum=W(k,heav,pos_x[:,t_],pos_x[:,t_+ti],pos_y[:,t_],pos_y[:,t_+ti])       
            W_sum_=W(-k,heav,pos_x[:,t_],pos_x[:,t_+ti],pos_y[:,t_],pos_y[:,t_+ti])
            W_init=W_sum+W_init                                                                      
            W_init_=W_sum*W_sum_+W_init_
            loop_n=loop_n+1
            
    S4_q_t.append([ti/fps,((W_init_/loop_n)-((W_init/loop_n)**2))/N])               # Calculating S4 using equation 7 of Analysis of a growing dynamics length scale paper
    chi4_t.append([ti/fps,((N_s_init/loop_n)-((N_s_init_/loop_n)**2))/N])           # Calculating chi_4 using formula 10 in same paper
    eta_t.append([ti/fps,(((((N_s_init/loop_n)-((N_s_init_/loop_n)**2))/N)/(((W_init_/loop_n)-((W_init/loop_n)**2))/N))-1)/k**2])           #Calculating zeta(and not eta) using equation 16 of same paper

print(np.shape(eta_t))

for i in range(0, np.shape(eta_t)[0]):
    print(i,"th zeta = ", eta_t[i][1])

# S4_q_t = np.array(S4_q_t)
# chi4_t = np.array(chi4_t)
# eta_t = np.array(eta_t)

# # print(S4_q_t)

# np.save('S4_q_t_'+str(conc),S4_q_t)
# np.save('chi4_t_'+str(conc),chi4_t)
# np.save('eta_t_'+str(conc),eta_t)

# end=time.time()

# print('Runtime : ',end-start)




# #plotting S4_q_t

# plt.title('Four point Correlation function')
# plt.xscale('log')
# plt.yscale('log')
# plt.axis('equal')
# plt.ylabel('S4_q_t')
# plt.xlabel('lag time $t$')
# plt.plot(S4_q_t[:,0],S4_q_t[:,1])
# plt.show()

# #plotting chi4_t

# plt.title('Dynamic Susceptibility')
# plt.xscale('log')
# plt.yscale('log')
# plt.axis('equal')
# plt.ylabel('chi4_t')
# plt.xlabel('lag time $t$')
# plt.plot(chi4_t[:,0],chi4_t[:,1])

# plt.show()

# #plotting eta_t

# plt.title('Dynamic correlation length')
# plt.xscale('log')
# plt.yscale('log')
# plt.axis('equal')
# plt.ylabel('eta_t')
# plt.xlabel('lag time $t$')
# plt.plot(eta_t[:,0],eta_t[:,1])

# plt.show()


