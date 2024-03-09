# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:44:04 2024

@author: berej
"""


import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.special as sci
from scipy.integrate import quad
from scipy.integrate import fixed_quad
from scipy.integrate import trapezoid
from scipy.integrate import romberg
from scipy.integrate import simpson
import scipy

import scienceplots

plt.style.use(['science', 'ieee'])

plt.close('all')

#initial params

#we take theta pi/2 but you can average over the solid angle instead.

theta = np.pi/2 #degrees
sintheta = np.sin(theta)

k = 1
p = 2
B = 500 #Gauss

sigma_T = 6.65e-25
c = 3.00e+10
B = 500
u_B = B*B/(8.0*np.pi)
m_e = 9.11e-28

A = (4/3 * sigma_T * u_B)/(m_e * c)

e = 4.8e-10
gmin = 100 #100 to see the fast cooling case, otherwise set this to 10
gmax = 1e3
g0 = 800

nu_min = (e*B)/(2*np.pi*m_e*c) * gmin**2
nu_max = (e*B)/(2*np.pi*m_e*c) * gmax**2

def t_syn(gamma):
    return (gamma*m_e*c)/(4/3 * sigma_T * u_B * gamma**2)

def gamma_0(t):
    return (m_e*c)/(4/3 * sigma_T * u_B * t)

def Bessel5_3(x):
    return sci.kv(5/3, x)
    
def nu_s(g):
    return g**2 * (e*B)/(m_e*c)
    
def nu_c(g):
    return 3/2 * nu_s(g) * sintheta

def F(x, g):
    
    res = []
    nuc = nu_c(g)
    
    
    if isinstance(x, np.ndarray):
        for n in x:
            res.append(n/nuc * quad(Bessel5_3, n/nuc, np.inf)[0])
            
        return np.array(res)
    
    else:
        return x/nuc * quad(Bessel5_3, x/nuc, np.inf)[0]
    
def Fg(x, g): #the actual function used, the previous one was included for testing purposes
    
    res = []
    
    for gam in g:
        res.append(x/nu_c(gam) * quad(Bessel5_3, x/nu_c(gam), np.inf)[0])
    
    return np.array(res)

    
nu_nuc = np.logspace(-3, 1, 100)
nu = np.logspace(np.log10(nu_min)-2, np.log10(nu_max)+2, 100)
gamma = np.logspace(-1, 4, 100)
N = np.zeros(len(gamma))

J_syn = []
Ng = []
t_array = []

fig, axs = plt.subplots(1, 2, num='constant injection', figsize=(15, 15), constrained_layout=True)


#this section generates the colorbar
n_lines = 10
cl = np.arange(0, 110, 10)
norm = mpl.colors.Normalize(vmin=cl.min(), vmax=cl.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
cmap.set_array([])

for t in range(0, 100, 10):
    for i in range(len(gamma)):
        if gamma[i] >= g0/(1+g0*t*A) and gamma[i] < g0:
            N[i] = gamma[i]**(-2)/A
    axs[0].plot(gamma, N, alpha = (10+t)/100, linestyle = 'solid', c=cmap.to_rgba(t))
    for n in nu:
        print('at nu {nu_text:2.2e}'.format(nu_text=n))
        J_syn.append(simpson(N*Fg(n, gamma), gamma))
    axs[1].plot(nu, np.array(J_syn), label = 'Time = {}'.format(t), alpha = (10+t)/100, linestyle = 'solid', c=cmap.to_rgba(t))
    Ng.append(simpson(N, gamma))
    t_array.append(t)
    N = np.zeros(len(gamma))
    J_syn = []


axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel(r'$\log{\nu}$ [Hz]')
axs[1].set_ylabel(r'$\log{J_{syn}}$ [erg/s/Hz]')
axs[1].set_ylim((1e-2, 1e2))

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlim((10, 1e4))
axs[0].set_xlabel(r'$\log{\gamma}$')
axs[0].set_ylabel(r'$\log{N(\gamma)}$')
axs[0].plot(gamma, 5e3*gamma**-2, label=r'$\gamma^{-2}$')
axs[0].legend(fontsize=5)
cbar = fig.colorbar(cmap, ticks = cl, label = 'Time (seconds)', orientation ='horizontal', ax=axs.ravel().tolist())


#uncomment this for the number of particles w.r.t time
# =============================================================================
# fig, axs = plt.subplots(num='constant injection N(t)_p{}'.format(p))
# 
# axs.plot(t_array, Ng, label = 'Numerical Integration')
# axs.plot([t_array[0], t_array[-1]], [t_array[0], t_array[-1]], label = 'Analytical solution', linestyle = 'dashed')
# #axs.set_xscale('log')
# #axs.set_yscale('log')
# axs.set_xlabel(r'$\log{t}$')
# axs.set_ylabel(r'$\log{N(t)}$')
# axs.set_title('p = {}'.format(p))
# axs.legend()
# =============================================================================

Ng = []
t_array = []

fig, axs = plt.subplots(1, 2, num='t0 power law', figsize=(15, 15), constrained_layout=True)

n_lines = 10
cl = np.arange(0, 1000, 100)
norm = mpl.colors.Normalize(vmin=cl.min(), vmax=cl.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
cmap.set_array([])

for t in range(0, 1000, 100):
    print("at time step ", t, " gmax bound ", gmax/(1+A*gmax*t))
    print("at time step ", t, " gmin bound ", gmin/(1+A*gmin*t))
    for i in range(len(gamma)):
        gstar = gamma[i]/(1-gamma[i]*A*t)
        #if gmax >= gstar and gstar >= gmin
        if gmax/(1+A*gmax*t) >= gamma[i] and gamma[i] >= gmin/(1+A*gmin*t):
            N[i] = gamma[i]**(-2) * gstar**(2-p)
    axs[0].plot(gamma, N, alpha = (100+t)/1000, label = 'N, Time = {}'.format(t), linestyle = 'solid', c=cmap.to_rgba(t))
    #if t != 0: #uncomment this and the following line to plot the equivalent gamma_0 for a given t_syn at the current timestep
    #    axs[0].plot([gamma_0(t), gamma_0(t)], [min(N), max(N)], color = 'black', linestyle = 'dotted')
    for n in nu:
        print('at nu {nu_text:2.2e}'.format(nu_text=n))
        J_syn.append(simpson(N*Fg(n, gamma), gamma))
    axs[1].plot(nu, np.array(J_syn), label = 'Time = {}'.format(t), alpha = (100+t)/1000, linestyle = 'solid', c=cmap.to_rgba(t))
    Ng.append(simpson(gamma*N, np.log(gamma)))
    t_array.append(t)
    N = np.zeros(len(gamma))
    J_syn= []
    
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel(r'$\log{\nu}$ [Hz]')
axs[1].set_ylabel(r'$\log{\nu J_{syn}}$ [erg/s]')


axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlim((1e0, 1e3))
axs[0].set_xlabel(r'$\log{\gamma}$')
axs[0].set_ylabel(r'$\log{\gamma^2 N(\gamma)}$')
cbar = fig.colorbar(cmap, ticks = cl, label = 'Time (seconds)', orientation ='horizontal', ax=axs.ravel().tolist())

fig, axs = plt.subplots(num='N(t)_p{}'.format(p))

axs.plot(t_array, Ng, label = 'Numerical Integration')
axs.plot([t_array[0], t_array[-1]], [1/(1-p) * (gmax**(1-p) - gmin**(1-p)), 1/(1-p) * (gmax**(1-p) - gmin**(1-p))], label = 'Analytical solution', linestyle = 'dashed')
axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlabel(r'$\log{t}$')
axs.set_ylabel(r'$\log{N(t)}$')
axs.set_title('p = {}'.format(p))
axs.legend()

Ng = []
t_array = []

fig, axs = plt.subplots(1, 2, num='slow cooling', figsize=(15, 15), constrained_layout=True)

n_lines = 10
cl = np.arange(0, 55, 5)
norm = mpl.colors.Normalize(vmin=cl.min(), vmax=cl.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
cmap.set_array([])

t_arr_n = np.logspace(0, 1.7, 10)

for t in t_arr_n:
    for i in range(len(gamma)):
        
        if gmax/(1+A*gmax*t) >= gmin: #slow cooling
            if gmax/(1+A*gmax*t) >= gamma[i] and gamma[i] >= gmin:
                N[i] = gamma[i]**(-(p+1)) / (A*(p-1)) * (1 - (1 - A*gamma[i]*t)**(p-1))
            if gmax/(1+A*gmax*t) < gamma[i] and gamma[i] <= gmax:
                N[i] = gamma[i]**(-2) / (A*(p-1)) * (gamma[i]**(-p+1) - gmax**(-p+1))
        else: #fast cooling
            if gamma[i] >= gmax/(1+A*gmax*t) and gamma[i] < gmin:
                N[i] = (gamma[i])**(-2)/(A*gmin*(p-1))
            if gamma[i] >= gmin and gamma[i] <= gmax:
                N[i] = gamma[i]**(-2) / (A*(p-1)) * (gamma[i]**(-p+1) - gmax**(-p+1))
                
    axs[0].plot(gamma, gamma**p*N, alpha = 0.5, label = 'Time = {}'.format(t), linestyle = 'solid', c=cmap.to_rgba(t))
    
    for n in nu:
        print('at nu {nu_text:2.2e}'.format(nu_text=n))
        J_syn.append(simpson(N*Fg(n, gamma), gamma))
    axs[1].plot(nu, nu*np.array(J_syn), label = 'Time = {}'.format(t), alpha = 0.5, linestyle = 'solid', c=cmap.to_rgba(t))
    Ng.append(simpson(gamma*N, np.log(gamma)))
    t_array.append(t)
    N = np.zeros(len(gamma))
    J_syn = []


axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel(r'$\log{\nu}$ [Hz]')
axs[1].set_ylabel(r'$\log{\nu J_{syn}}$ [erg/s]')


axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlim((10, 1e4))
axs[0].set_xlabel(r'$\log{\gamma}$')
axs[0].set_ylabel(r'$\log{\gamma^p N(\gamma)}$')
cbar = fig.colorbar(cmap, ticks = cl, label = 'Time (seconds)', orientation ='horizontal', ax=axs.ravel().tolist())

tarr = np.array(t_array)

#uncomment this for the number of particles w.r.t time
# =============================================================================
# fig, axs = plt.subplots(num='slow cooling N(t)_p{}'.format(p))
# 
# axs.plot(t_array, Ng, label = 'Numerical Integration')
# #axs.plot(tarr, 1/(A*(p-1)*p) * (1/gmin**p - ((1-A*np.array(t_array)*gmin)/gmin)**p - (2+A*np.array(t_array)*p*gmax)/gmax**p), label = 'Analytical solution', linestyle = 'dashed')
# #axs.plot(tarr, 1/(A*(p-1)*p) * ((1-(1-A*tarr*gmin)**p)/gmin**p - (gmax**p - (gmax/(1+A*tarr*gmax))**p)/(gmax**p * (gmax/(1+A*tarr*gmax))**p ) + ((1+A*tarr*gmax)**p - 1 - A*p*tarr*gmax)/gmax**p))
# axs.plot(tarr, 1/(A*(p-1)*p) * ((-gmax**p * (1-A*gmin*tarr)**p + gmin**p + gmax**p)/(gmax**p * gmin**p) - 1/(gmax/(A*gmax*tarr+1))**p + ((A*tarr*gmax+1)**p - A*gmax*p*t-1)/gmax**p))
# axs.set_xscale('log')
# axs.set_yscale('log')
# axs.set_xlabel(r'$\log{t}$')
# axs.set_ylabel(r'$\log{N(t)}$')
# axs.set_title('p = {}'.format(p))
# axs.legend()
# 
# =============================================================================
