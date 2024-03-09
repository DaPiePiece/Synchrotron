# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:14:23 2024

@author: berej
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.special as sci
from scipy.integrate import quad

import scienceplots


#Initial parameters in cgs units

k = 1
p = 2
B = 500 #Gauss
e = 4.8e-10
m_e = 9.11e-28
c = 3e10 #cm/s
theta = np.pi/2 #degrees
sintheta = np.sin(theta)
gmin = 1
gmax = 1e3

nu_min = (e*B)/(2*np.pi*m_e*c) * gmin**2
nu_max = (e*B)/(2*np.pi*m_e*c) * gmax**2

def Bessel5_3(x):
    return sci.kv(5/3, x)
    
def nu_s(g):
    return g**2 * (e*B)/(m_e*c)
    
def nu_c(g):
    return 3/2 * nu_s(g) * sintheta

def Ng(g):
    return k * g**(-p)

def F(x, g):
    
    res = []
    nuc = nu_c(g)
    
    
    if isinstance(x, np.ndarray):
        for n in x:
            res.append(n/nuc * quad(Bessel5_3, n/nuc, np.inf)[0])
            
        return np.array(res)
    
    else:
        return x/nuc * quad(Bessel5_3, x/nuc, np.inf)[0]
    
def F_apr(x, g): #approximation
    return (4*np.pi)/(np.sqrt(3)*sci.gamma(1/3)) * (1/2 * x/nu_c(g))**(1/3) * np.exp(-x/nu_c(g))


#comment these lines out if you don't use scienceplots, but I recommend getting scienceplots.
plt.style.use(['science', 'ieee'])

plt.close('all')

nu_nuc = np.logspace(-3, 1, 100)
nu = np.logspace(np.log10(nu_min)-2, np.log10(nu_max)+2, 100)
theor = nu**((1-p)/2)
gamma = np.logspace(gmin, gmax, 100)
J_syn = []

#quad takes a bit of time.

for n in nu:
    print('at nu', n)
    J_syn.append(quad(lambda g: Ng(g)*F(n, g), gmin, gmax)[0])


fig, axs = plt.subplots(num='nu nuc comparison')

axs.plot(nu/nu_c(10000), F(nu, 10000), label = 'gamma = 10000')
axs.plot(nu/nu_c(1000), F(nu, 1000), label = 'gamma = 1000')
axs.plot(nu/nu_c(100), F(nu, 100), label = 'gamma = 100')
axs.plot(nu/nu_c(100), F_apr(nu, 100), label = 'gamma = 100, approximation')

axs.legend()

axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlabel(r'$\log{\nu/\nu_c}$')
axs.set_ylabel(r'$\log{F\left(\nu/\nu_c\right)}$')
axs.set_xlim((0.001, 10))
axs.set_ylim((0.01, 5))

fig, axs = plt.subplots(num = 'Full spectrum')

axs.plot(nu, np.array(J_syn)/max(J_syn), label = 'integrated J_syn')
axs.plot(nu, 5*np.array(theor)/max(theor), label = 'Theoretical power law')

axs.legend()

axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlabel(r'$\log{\nu}$ [Hz]')
axs.set_ylabel(r'$\log{J_{syn}}$ [erg/Sr/s/Hz]')
