# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:28:43 2024

@author: berej
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.special as sci
from scipy.integrate import quad


import scienceplots

plt.style.use(['science', 'ieee'])

plt.close('all')


theta = np.pi/2 #degrees
sintheta = np.sin(theta)

k = 1
p = 2
#B = 500 #Gauss
#B = 6.00689847e-01 #Gauss, Scipy lsqrs
#B = 3.64887778e-01 #Gauss, ODR whole fit
B = 5.06522862e-01 #Gauss, ODR partial fit


sigma_T = 6.65e-25
c = 3.00e+10
#B = 500
u_B = B*B/(8.0*np.pi)
m_e = 9.11e-28

A = (4/3 * sigma_T * u_B)/(m_e * c)

e = 4.8e-10
#gmin = 10
gmin = 121016.27783628 #lsqrs with fixed params
#gmax = 1e3
#gmax = 1.10624578e+06 #Scipy lsqrs
#gmax = 1.73047029e+06 #Scipy ODR whole
gmax = 1.45827796e+06 #Scipy ODR partial
g0 = 800
D = 350*3.086e+24 #cm

nu_min = (e*B)/(2*np.pi*m_e*c) * gmin**2
nu_max = (e*B)/(2*np.pi*m_e*c) * gmax**2

#delta = 3.70752427e+02 #Scipy lsqrs
#delta = 1.89525501e+03 #Scipy ODR
delta = 2.88869280e+02 #Scipy ODR partial

erg_to_kev = 6.242e+8
h = 6.6261e-27
q0_2 = 1.5e25
q6_8 = 8e25
q10_12 = 4e24
q18_20 = 3e24
q30_34 = 1.2e24
q50_60 = 0.3e24

timeframes = [(0, 2), (6, 8), (10, 12), (18, 20), (30, 34), (50, 60)]
q0s = [q0_2, q6_8, q10_12, q18_20, q30_34, q50_60]

def t_syn(gamma):
    return (gamma*m_e*c)/(4/3 * sigma_T * u_B * gamma**2)

def gamma_0(t):
    return (m_e*c)/(4/3 * sigma_T * u_B * t)

def Bessel5_3(x):
    return sci.kv(5/3, x)
    
def nu_s(g):
    return g**2 * (e*B)/(2*np.pi*m_e*c)
    
def nu_c(g):
    return 3/2 * nu_s(g) * sintheta

def Ng_func(g):
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
    
def Fg(x, g):
    
    res = []
    
    for gam in g:
        res.append(x/nu_c(gam) * quad(Bessel5_3, x/nu_c(gam), np.inf)[0])
    
    return np.array(res)

def gamma_to_kev(gamma, B=B, delta=delta):
    nu_s = e*B * gamma**2 / (2*np.pi*m_e*c)
    return nu_s*h*delta*erg_to_kev

def gstar(gamma, t, delta=delta):
    t_rf = t*delta

    return gamma/(1+A*t_rf*gamma)

    
nu_nuc = np.logspace(-3, 1, 100)
nu = np.logspace(np.log10(nu_min)-2, np.log10(nu_max)+2, 100)
gamma = np.logspace(np.log10(gmin)-2, np.log10(gmax)+2, 100)
N = np.zeros(len(gamma))

J_syn = []
Ng = [] 

start = 0
stop = 120
step = 5

t_array = np.linspace(6, 8, 10)
J_array = []

fig, axs = plt.subplots(num='t0-2')


cl = np.arange(0, 130, 10)

norm = mpl.colors.Normalize(vmin=cl.min(), vmax=cl.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
cmap.set_array([])

q=1

for t in t_array:
    t *= delta
    for i in range(len(gamma)):
        
        if gmax/(1+A*gmax*t) >= gmin: #slow cooling
            if gmax/(1+A*gmax*t) >= gamma[i] and gamma[i] >= gmin:
                N[i] = q0s[q]*gamma[i]**(-(p+1)) / (A*(p-1)) * (1 - (1 - A*gamma[i]*t)**(p-1))
            if gmax/(1+A*gmax*t) < gamma[i] and gamma[i] <= gmax:
                N[i] = q0s[q]*gamma[i]**(-2) / (A*(p-1)) * (gamma[i]**(-p+1) - gmax**(-p+1))
        else: #fast cooling
            if gamma[i] >= gmax/(1+A*gmax*t) and gamma[i] < gmin:
                N[i] = q0s[q]*(gamma[i])**(-2)/(A*gmin*(p-1))
            if gamma[i] >= gmin and gamma[i] <= gmax:
                N[i] = q0s[q]*gamma[i]**(-2) / (A*(p-1)) * (gamma[i]**(-p+1) - gmax**(-p+1))

    t *= 1/delta

    for n in nu:
        print('at nu {nu_text:2.2e}'.format(nu_text=n))
        J_syn.append(np.trapz(N*Fg(n, gamma), gamma))
    axs.plot(delta*nu*h*erg_to_kev, delta**4*nu*np.array(J_syn)/(4*np.pi*D**2), alpha = 0.5, linestyle = 'solid', c=cmap.to_rgba(t))

    J_array.append(delta**4*nu*np.array(J_syn)/(4*np.pi*D**2))
    N = np.zeros(len(gamma))
    J_syn = []

J_arr = np.array(J_array)
gmax_star_arr = gstar(gmax, t_array)
gmax_star_mean = np.average(gmax_star_arr)
axs.plot(delta*nu*h*erg_to_kev, np.average(J_arr, axis=0), label='6-8s average')
axs.plot([gamma_to_kev(gmin), gamma_to_kev(gmin)], [1e-8, 1e-4], label='Eb')
axs.plot([gamma_to_kev(gmax_star_mean), gamma_to_kev(gmax_star_mean)], [1e-8, 1e-4], label='Ep')
axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlim((1e1, 1e4))
axs.set_ylim((1e-8, 1e-4))
axs.set_xlabel(r'Observer Frame Energy [keV]')
axs.set_ylabel(r'$\log{\nu F_{\nu}}$ $[erg/cm^{-2}/s]$')

art_dat = np.loadtxt('./data/6-8s.txt', unpack=True, skiprows=1)
axs.errorbar(art_dat[0], art_dat[2]/erg_to_kev, xerr=art_dat[1], yerr=art_dat[3]/erg_to_kev, markersize=2, marker=".", ls='none', label='Article Data', alpha=0.5)
axs.legend()
