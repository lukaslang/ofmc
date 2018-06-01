#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2017 Lukas Lang
#
# This file is part of OFMC.
#
#    OFMC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    OFMC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with OFMC.  If not, see <http://www.gnu.org/licenses/>.
import os
import datetime
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ofmc.model.cmscr import cmscr1d_img
from scipy.sparse import spdiags
import scipy.stats as stats

# Set font style.
font = {'weight': 'normal',
        'size': 20}
plt.rc('font', **font)

# Set colormap.
cmap = cm.viridis

# Streamlines.
density = 2
linewidth = 2

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)


def savequantity(path: str, name: str, img: np.array, title: str):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cmap)
    ax.set_title(title)

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def saveimage(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cm.gray)
    ax.set_title('Fluorescence intensity')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def savesource(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cmap)
    ax.set_title('Source')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-source.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def savevelocity(path: str, name: str, img: np.array, vel: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    maxvel = abs(vel).max()
    normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(vel, interpolation='nearest', norm=normi, cmap=cmap)
    ax.set_title('Velocity')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-velocity.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    m, n = vel.shape
    hx, hy = 1.0 / (m - 1), 1.0 / (n - 1)

    # Create grid for streamlines.
    Y, X = np.mgrid[0:m, 0:n]
    V = np.ones_like(X)

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    ax.set_title('Streamlines')
    strm = ax.streamplot(X, Y, vel * hx / hy, V, density=density,
                         color=vel, linewidth=linewidth, norm=normi, cmap=cmap)
#    fig.colorbar(strm.lines, orientation='vertical')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(strm.lines, cax=cax, orientation='vertical')

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-streamlines.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save velocity profile after cut.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(vel[5])
    ax.set_title('Velocity profile right after the cut')

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-profile.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def savestrainrate(path: str, name: str, img: np.array, vel: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    m, n = img.shape
    hy = 1.0 / (n - 1)
    sr = np.gradient(vel, hy, axis=1)

    maxsr = abs(sr).max()
    normi = mpl.colors.Normalize(vmin=-maxsr, vmax=maxsr)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(sr, interpolation='nearest', norm=normi, cmap=cmap)
    ax.set_title('Strain rate')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-strainrate.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

##################################
#### PHYSICAL PARAMETERS #########
##################################

T_final =0.1
T_cut = 0.01

#inital conditions
L             =1.0 # curvilinear length between the two poles  
def ca_handle(x):  		return  stats.uniform.pdf(x, 0, L)*20   #initial distribution of attached motors
def cd_handle(x): 		return  stats.uniform.pdf(x, 0, L)*20   #initial distribution of dettached motors
def rho_handle(x): 		return stats.uniform.pdf(x, 0, L)*20   #initial distribution of actin
def a_handle(x): 		return stats.uniform.pdf(x, 0, L)*20+ 10*np.sin(100*L*x+ np.cos(100*L*x))/(0.5*L)  #initial distribution of features. 

#################################
## ART. VEL. PARAMETERS #########
#################################
artvel=False #Switch ON/OFF ARTIFICIAL VELO FIELD

tau_t=10 #coefficient of exponent for cut
v_0=5.1 #velocity at cut 'lips'
a_0=2.1 #initial cut width

#################################
## MECHANICAL PARAMETERS ########
#################################
eta           =0.5            # viscosity of the cortex 
xi            =0.15         # friction with the (sort of static ?) membrane  
E             =1.0            # elastic modulus of the actine meshwork (or compressibility)         
chi           =0.5          # contractility modulus
Diffuca        =0.0            # diffusion coefficient of attached mysoin
Diffucd        =0.0           # diffusion coefficient of the detached myosin
Diffurho      =0.0000            # diffusion coefficient of actin (interpretation as a non local source term)
rho_e         =1.0            # 'reference' actin density of elasticity
c_c           =1.0            # 'reference' myosin concentration of contraction


# chemistry ###################
rho_0         =1.0            # target density of actin
tau           =1.0            # tunover time of actin

def k_on(y):  		return   1.000  #+1.0*exp(-(y-0.5).^2/0.05)                    # rate of myosin attachement (here centered at the equator)
def k_off(y):  		return   3.0 #*(exp(-y.^2/0.05)+exp(-(y-1.0).^2/0.05))  # rate of myosin detachement (here high at the poles)

#mechanical noise (small biological gaussian noise in the force equation).
# D_m         =1.0e-8         # Small Gaussian noise on c
D_m = 0

#################################
### RHEOLOGY FUNCTIONS  #########
#################################

def f_e(rho):  			return  0           #rho/rho_e # elastic  compressibility, here infinite compressibility (f_e=0)
def f_c(rho, c):  		return    c/c_c     # contraction (here only depending on attached motors)
def S_rho(rho):  		return (rho_0-rho)/tau        # source term of actin 
def ytox(y,r,f):  		return  y*(f-r)+r  


#############################
### NUMERICAL PARAMETERS ####
#############################
dt_jump=100                    # number of jumped snapshots of the solution before one is save
N=101                         # number of space points
N_h=int(N/2.0)	 
CFL=0.98                       # CFL condition
dt_min=1.0e-5                    # Minimal timestep
Nsteps_jumped= int(10*T_final*1.0/(dt_min*dt_jump))            # Maximal number of stored time steps 
delta=0.001                    # Small parameter for the entropic remede in case of Gibbs oscillations in shocks 

def f_ent(x):  		return (x*(x>delta)+(x**2/(2*delta)+delta/2)*(x<delta)) # Function for the entropic Remede (See Allaire Cocquel)

print('Number of space points: N=', N)
print('NSteps_jumped= ', Nsteps_jumped)

k=2 #length of the cut in units of number of points
f_cut=np.ones(N) #cutting func
f_cut[(N-1)//2-k:(N-1)//2+k]=0 #cutting func contd. 




########################
### ALLOCATION #########
########################

dy=1./(N-1)                            # Spacing of the grid
 

Y=np.linspace(dy/2,1-dy/2,num=N)
Y=Y.transpose()            			#defining the grid 
Yd=np.linspace(0,1,num=N+1)                  # defining the dual grid


Sig=np.zeros(N+2)                         #Stress
Sig_sav=np.zeros((N+2,Nsteps_jumped))         #saved stress

v=np.zeros(N+1)                           #Velocity
v_sav=np.zeros((N+1,Nsteps_jumped))           # saved velocity

caL=np.zeros((N,2))                         # Density of attached myosin mulptipilied by L 
ca_sav=np.zeros((N,Nsteps_jumped))           # saved attached myosin concentration

cdL=np.zeros((N,2))                         # Density of detached myosin mulptipilied by L 
cd_sav=np.zeros((N,Nsteps_jumped))           # saved detached myosin concentration

rhoL=np.zeros((N,2))                         # Density of actin mulptipilied by L 
rho_sav=np.zeros((N,Nsteps_jumped))           # saved actin density


aL=np.zeros((N,2))                         # features, mulptipilied by L 
a_sav=np.zeros((N,Nsteps_jumped))            # saved features



TimeS=np.zeros((1,Nsteps_jumped))          # saved time steps
r=np.zeros((1,2))                          #rear
r_sav=np.zeros((1,Nsteps_jumped))
f=np.zeros((1,2))                          #front
f_sav=np.zeros((1,Nsteps_jumped))


Fa=np.zeros(N+1)                        # Flux of attached myosin 
Fd=np.zeros(N+1)                        # Flux of detached myosin 
Frho=np.zeros(N+1)                      # Flux of actin 
Faf=np.zeros(N+1)         
Fav=np.zeros(N+1)         
one=np.ones(N)                          # Macro


################################################
### INITIALIZATION OF THE SOLUTION AT T=0 ######
################################################

Time=0.0                               #actual time
TimeS[0,0]=0.0                         #Saved times

ii = 0
j=1

                 
ca_sav[:,0]=ca_handle(ytox(Y,0,L))
cd_sav[:,0]=cd_handle(ytox(Y,0,L))
rho_sav[:,0]=rho_handle(ytox(Y,0,L))
a_sav[:,0]=a_handle(ytox(Y,0,L))

caL[:,0]=ca_sav[:,0]*L      #initial concentration for detached myosin times L 
cdL[:,0]=cd_sav[:,0]*L        #initial concentration for detached myosin times L 
rhoL[:,0]=rho_sav[:,0]*L        #initial density for attached myosin times L 
aL[:,0]=a_sav[:,0]*L        #initial concentration for features times L 

Jac=L*dy
Jacc=Jac**2

data= np.array([-one,(2+xi*Jacc/eta)*one,-one])
diags= np.array([-1,0,1])
R=spdiags(data,diags, N, N).toarray() #Tridiagonal matrix
R[0,0]=1+xi*Jacc/eta
R[N-1,N-1]=1+xi*Jacc/eta



stres=np.zeros(N)


stres=  ( (xi*Jacc/eta) * chi*f_c(rhoL[:,0]/L,caL[:,0]/L) ) +np.divide(D_m/np.sqrt(Jac),np.random.randn(N))

temp=np.linalg.solve(R, stres ) # Computation of the stress inside the layer. 

Sig[1:N+1]= temp[0:N]

Sig[0]=Sig[1]
Sig[N+1]=Sig[N] #Computation of stress in side the cell


Sig_sav[:,0]=Sig


v=np.divide(np.diff(Sig, axis=0),(Jac*xi))              #Computation of velocity                     
v_sav[:,0]=v[:]


# computation of the drift to solve the transport equations 
drift_eff=v

# Chosing the time step according to the CFL condition

dt=np.min([CFL*Jac/np.max(np.abs(drift_eff)+2*np.max([Diffuca,Diffucd,Diffurho])/Jac),dt_min])    
print('dt: t=',dt)


###################################
### GENERAL TIME STEPPING #########
###################################


plt.ion()


while Time<= T_final:
	try:
	
		Time = Time + dt
		j=j+1

		#########################################	
		#######ARTIFICIAL VELOCITY FIELD ########
		if(artvel): 
			ex=np.exp(-Time*1./ tau_t)

			v_a=(v_0*ex)
			a_a=(a_0+tau_t*v_0*(1-ex))/(a_0+tau_t*v_0)
			n_a_a=int(a_a*(N+1))

			if(n_a_a >= N_h): 
				print ('cut too big for simulation domain')
				break

			if(n_a_a < 2):
				print ('cut too small for simulation domain')
				break

			grid1= N_h-n_a_a
			grid2= N_h+n_a_a+1 
	
			point1= grid1*1.0/(N+1)
			point2= grid2*1.0/(N+1)

		
			ex1=(np.log(v_a)+10.0)/point1	
			ex2=(10.0+np.log(v_a))/(1-point2)	
		
			v[0:N_h]= -np.exp( ex1*Yd[0:N_h]-10.0 ) 
			v[N_h+1:N+1]= np.exp(-(ex2*Yd[N_h+1:N+1]- (ex2-10.0) ))		
			v[grid1:grid2+1]= np.linspace( -v_a, v_a, 2*n_a_a+2)
			drift_eff=v

			caL[grid1:grid2+1,0]= 0.0;
			cdL[grid1:grid2+1,0]= 0.0;
			aL[grid1:grid2+1,0]= 0.0;
						
		#########################################
		#########################################

		

		drift_ent=f_ent(np.abs(drift_eff[1:N]))       #Remede entropique pour choc stationnaire (cours Allaire-Coquel)
		Fa[1:N]=0.5*( np.multiply( drift_eff[1:N],(caL[1:N,0]+caL[0:N-1,0]) )-np.multiply( (drift_ent+2*Diffuca/Jac),(caL[1:N,0]-caL[0:N-1,0]) ) )
		Fd[1:N]=np.multiply( (-Diffucd/Jac),(cdL[1:N,0]-cdL[0:N-1,0]) )

		Frho[1:N]=0.5*( np.multiply( drift_eff[1:N],(rhoL[1:N,0]+rhoL[0:N-1,0]) )-np.multiply( (drift_ent+2*Diffurho/Jac),(rhoL[1:N,0]-rhoL[0:N-1,0]) ) )

		Faf[1:N]=0.5*( np.multiply( drift_eff[1:N],(aL[1:N,0]+aL[0:N-1,0]) ) )	
		Fav[0:N+1]=( drift_eff[0:N+1] )
		 
		 
		#We impose no flux BC on actin
		Frho[0]=0.0
		Frho[N]=0.0 

		#We impose no flux BC on motors
		Fa[0]=0.0
		Fa[N]=0.0   

		Fd[0]=0.0
		Fd[N]=0.0 
		#We impose no flux BC on features
		Faf[0]=0.0
		Faf[N]=0.0 

	
		#########################################
		####################CUT #################
		if(not artvel):
			if(Time>=T_cut-1.0*dt and Time<T_cut+1.0*dt ) :
		
				caL[:,0]= np.multiply(caL[:,0],f_cut)
				aL[:,0]= np.multiply(aL[:,0],f_cut)
		
		##########################################
		##########################################
	


		
		rhoL[:,1]=rhoL[:,0]  # actin - here the dynamics of actin have been removed. 

		cdL[:,1]=cdL[:,0]-dt*(np.diff(Fd,axis=0)/Jac)-L*( np.multiply(k_on(Y),cdL[:,0]/L)-np.multiply(k_off(Y),caL[:,0]/L) )*dt # detached motors 
		caL[:,1]=caL[:,0]-dt*(np.diff(Fa, axis=0)/Jac)+L*( aL[:,0]*1.0/(L*1.0) - np.multiply(k_off(Y),caL[:,0]/L)  )*dt # attached motors 

		aL[:,1]=aL[:,0]-dt*(np.diff(Faf, axis=0)/Jac)+( np.multiply(aL[:,0],np.diff(Fav, axis=0)/Jac) )*dt

	
		#########################################	
		###########MECHANICS#####################
		if(not artvel): 
	
			stres=np.zeros(N)
			stres=  ( (xi*Jacc/eta) * chi*f_c(rhoL[:,0]/L,caL[:,0]/L) ) +np.divide(D_m/np.sqrt(Jac),np.random.randn(N))
			temp=np.linalg.solve(R, stres ) # Computation of the stress inside the layer. 

			Sig[1:N+1]= temp[0:N]
			Sig[0]=Sig[1]
			Sig[N+1]=Sig[N] #Computation of stress in side the cell
	


			v=np.divide(np.diff(Sig, axis=0),(Jac*xi))
		#########################################
		#########################################

		############################
		###Saving some time steps###
		############################
		  
		if (j%dt_jump==0) :
			ii = ii + 1
			v_sav[:,ii]=v[:] # saved actin velocity
			Sig_sav[:,ii]=Sig[:] # saved actin meshwork stress
			rho_sav[:,ii]=rhoL[:,1]/L # saved density of actin
			ca_sav[:,ii]=caL[:,1]/L # saved concentration of attached motors
			cd_sav[:,ii]=cdL[:,1]/L # saved concentration of detached motors
			a_sav[:,ii]=aL[:,1]/L #saved features
			TimeS[0,ii]=Time # saved times
			print ('Time=', Time)
		
		
	
	 
		###################################
		### Preparing the next time step ##
		###################################
		 

		# effective flux
		drift_eff=v 

		# densities updates
		rhoL[:,0]=rhoL[:,1]
		caL[:,0]=caL[:,1]
		cdL[:,0]=cdL[:,1]
		aL[:,0]=aL[:,1]


		# adapting the time step with the CFL condition
		dt=np.min([ CFL*Jac/np.max(np.abs(drift_eff)+2*np.max([Diffuca,Diffucd,Diffurho])/Jac), dt_min ])

		######################################
		####finish if array bounds exceeded###
		######################################


		if(ii >= Nsteps_jumped-2) :
			print ("array bounds exceeded at time t=", T_final)
			break

	except  KeyboardInterrupt: break

##################################################
#######END OF TIME STEPPING#######################
#######PRINTING TO DATA FILE######################
print('Done!\n')

# Plot results.
rng = range(ii)

# Set name.
name = 'mechanical_model'

resfolder = os.path.join(resultpath, 'mechanical_model')
if not os.path.exists(resfolder):
    os.makedirs(resfolder)

# Plot and save figures.
savequantity(resfolder, '{0}-a_sav'.format(name),
             a_sav[:, rng].transpose(), 'a_sav')
savequantity(resfolder, '{0}-ca_sav'.format(name),
             ca_sav[:, rng].transpose(), 'ca_sav')
savequantity(resfolder, '{0}-cd_sav'.format(name),
             cd_sav[:, rng].transpose(), 'cd_sav')
savequantity(resfolder, '{0}-ca_sav+a_sav'.format(name),
             (cd_sav[:, rng] + a_sav[:, rng]).transpose(), 'ca_sav + a_sav')
savequantity(resfolder, '{0}-v_sav'.format(name),
             v_sav[:, rng].transpose(), 'v_sav')

# Set regularisation parameter.
alpha0 = 5e0
alpha1 = 1e-1
alpha2 = 1e-2
alpha3 = 1e-2
beta = 5e-4

# Define concentration.
# img = (ca_sav[:, 11:ii] + a_sav[:, 11:ii]).transpose()
# img = ca_sav[:, 1:ii].transpose()
img = ca_sav[:, 11:ii].transpose()

# Compute velocity and source.
vel, k = cmscr1d_img(img, alpha0, alpha1, alpha2, alpha3,
                     beta, 'mesh')

# Plot and save figures.
saveimage(resfolder, name, img)
savevelocity(resfolder, name, img, vel)
savesource(resfolder, name, k)
savestrainrate(resfolder, name, img, vel)
