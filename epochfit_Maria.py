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
from astropy import units as u
from astropy import constants as const

plt.close('all')
##### constants ##############
c=(const.c).cgs.value     #cm/s speed of light 
yr=(u.yr).to(u.s)               #seconds
kpc=(u.kpc).to(u.cm)              #cm
pc=(u.pc).to(u.cm)              #cm
m_p=(u.M_p).to(u.g)        #gr proton mass
m_e=(u.M_e).to(u.g)        #gr electron mass   
kb=(const.k_B).cgs.value
h = (const.h).cgs.value 
e=(const.e.gauss).value                 #statcoulomb
sigma_T=(const.sigma_T).cgs.value               
evtoerg = 1.6*10**(-12)
erg_to_kev = 6.242e+8
##### functions ##############


def t_syn(gamma, u_B):
    return (gamma*m_e*c)/(4/3 * sigma_T * u_B * gamma**2)

def Bessel5_3(x):
    return sci.kv(5/3, x)
    
def nu_s(g, B):
    return g**2 * (e*B)/(2*np.pi*m_e*c)
    
def nu_c(g, B):
    theta = np.pi/2 #degrees
    sintheta = np.sin(theta)
    return 3/2 * nu_s(g, B) * sintheta

def F(x, g, B):
    
    res = []
    nuc = nu_c(g, B)
    
    
    if isinstance(x, np.ndarray):
        for n in x:
            res.append(n/nuc * quad(Bessel5_3, n/nuc, np.inf)[0])
            
        return np.array(res)
    
    else:
        return x/nuc * quad(Bessel5_3, x/nuc, np.inf)[0]
    
def Fg(x, g):
    
    res = []
    
    for gam in g:
        res.append(x/nu_c(gam, B) * quad(Bessel5_3, x/nu_c(gam, B), np.inf)[0])
    
    return np.array(res)

def gamma_to_kev(gamma, B, delta):
    nu_s = e*B * gamma**2 / (2*np.pi*m_e*c)
    return nu_s*h*delta*erg_to_kev

def gstar(gamma, t, delta, A):
    t_rf = t*delta
    return gamma/(1+A*t_rf*gamma)

def Ne_g(gamma, gmin, gmax, p, q0,  t, delta, A):
    gmax_star = gstar(gmax, t, delta, A)
    t_rf = t*delta
    Ne = np.zeros(len(gamma))
    for i in range(len(gamma)):
        if gmax_star >= gmin:
            if gmax_star >= gamma[i] and gamma[i] >= gmin:
                Ne[i] = q0*gamma[i]**(-(p+1)) / (A*(p-1)) * (1 - (1 - A*gamma[i]*t_rf)**(p-1))
            if gmax_star < gamma[i] and gamma[i] <= gmax:
                Ne[i] = q0*gamma[i]**(-2) / (A*(p-1)) * (gamma[i]**(-p+1) - gmax**(-p+1))
        else:
            if gamma[i] >= gmax_star and gamma[i] < gmin:
                Ne[i] = q0*(gamma[i])**(-2)/(A*gmin*(p-1))
            if gamma[i] >= gmin and gamma[i] <= gmax:
                Ne[i] = q0*gamma[i]**(-2) / (A*(p-1)) * (gamma[i]**(-p+1) - gmax**(-p+1))
    return Ne        

############# Parameter Values ###############
D = 350*1e6*pc #cm
#B = 500 #Gauss
#B = 6.00689847e-01 #Gauss, Scipy lsqrs
#B = 3.64887778e-01 #Gauss, ODR whole fit
B = 5.06522862e-01 #Gauss, ODR partial fit
u_B = B*B/(8.0*np.pi)
A = (4/3 * sigma_T * u_B)/(m_e * c)
gmin = 121016.27783628 #lsqrs with fixed params
#gmax = 1.10624578e+06 #Scipy lsqrs
#gmax = 1.73047029e+06 #Scipy ODR whole
gmax = 1.45827796e+06 #Scipy ODR partial
#delta = 3.70752427e+02 #Scipy lsqrs
#delta = 1.89525501e+03 #Scipy ODR
delta = 2.88869280e+02 #Scipy ODR partial

nu_min = (e*B)/(2*np.pi*m_e*c) * gmin**2
nu_max = (e*B)/(2*np.pi*m_e*c) * gmax**2

    
nu_nuc = np.logspace(-3, 1, 100)
nu = np.logspace(np.log10(nu_min)-2, np.log10(nu_max)+2, 50)
gamma = np.logspace(np.log10(gmin)-2, np.log10(gmax)+2, 50)
N = np.zeros(len(gamma))

J_syn = []
Ng = [] 

p = 1.5
# q0_2 = 2e25
# q6_8 = 8e25*0.001
# q10_12 = 4e24
# q18_20 = 3e24
# q30_34 = 1.2e24
# q50_60 = 0.3e24
# q0_arr = [q0_2, q6_8, q10_12, q18_20, q30_34, q50_60]
q0_arr = []
timeframes = [(0, 2), (6, 8), (10, 12), (18, 20), (30, 34), (50, 60)]



#n_lines = 10
# cl = np.arange(t_array[0], t_array[len(t_array)-1], 1)
# norm = mpl.colors.Normalize(vmin=cl.min(), vmax=cl.max())
# cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
# cmap.set_array([])
titles = ['0-2 s', '6-8 s',  '10-12 s', '18-20 s', '30-34 s', '50-60 s']
labels = ['0-2s', '6-8s', '10-12s', '18-20s', '30-34s', '50-60s']

Flux_array = []
Flux_arr_mean = []
for j in range(0,len(timeframes)-1):
    t_array  = np.linspace(timeframes[j][0], timeframes[j][1], 10)
    art_dat = np.loadtxt('./data/'+labels[j]+'.txt', unpack=True, skiprows=1)
    print('Timeframe #', j)    

    for t in t_array:   
        print('Obs. time (s)', t)
        N=Ne_g(gamma, gmin, gmax, p, 1,  t, delta, A)
    
        for n in nu:
            J_syn.append(np.trapz(gamma*N*Fg(n, gamma), np.log(gamma)))   
        
        Flux_array.append(delta**4*nu*np.array(J_syn)/(4*np.pi*D**2)) # nu*F_nu (erg/s/cm^2)
        N = np.zeros(len(gamma))
        J_syn = []
        
    Flux_arr = np.array(Flux_array)
    Flux_arr_mean.append(np.average(Flux_arr, axis=0))   
    print('peak flux data', max( art_dat[2]/erg_to_kev))
    q0_arr.append(max( art_dat[2]/erg_to_kev)/ max(np.average(Flux_arr, axis=0)))
    print('Normalization', q0_arr)

########### plots ###############

j = 1
art_dat = np.loadtxt('./data/'+labels[j]+'.txt', unpack=True, skiprows=1)
mpl.rcParams.update({'font.size': 18})
fig, axs = plt.subplots(num='', figsize=(10,6))
axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlim((1e1, 1e4))
axs.set_ylim((1e-8, 5e-4))
axs.set_xlabel(r'Observer Frame Energy [keV]')
axs.set_ylabel(r'$\log{\nu F_{\nu}}$ $[erg/cm^{2}/s]$')
axs.text(20, 1e-4, titles[j], horizontalalignment = 'center', 
          fontsize = 14, bbox=dict(boxstyle="round",ec='k',
                    fc='white'))
axs.errorbar(art_dat[0], art_dat[2]/erg_to_kev, xerr=art_dat[1], yerr=art_dat[3]/erg_to_kev, markersize=4, marker=".", ls='none', label='Article Data', alpha=0.7)
axs.plot(delta*nu*h*erg_to_kev, np.multiply(Flux_arr_mean[j], q0_arr[j]), alpha = 1, linestyle = '-', label='average model') 


t_array  = np.linspace(timeframes[j][0], timeframes[j][1], 10)
gmax_star_arr = []
for t in t_array: 
    gmax_star_arr.append(gstar(gmax, t, delta, A))

gmax_star_mean = np.average(np.array(gmax_star_arr))
axs.plot([gamma_to_kev(gmax_star_mean, B, delta), gamma_to_kev(gmax_star_mean, B, delta)], [1e-8, 5e-4], label='Ep', color='grey')
axs.plot([gamma_to_kev(min(gmax_star_arr), B, delta), gamma_to_kev(min(gmax_star_arr), B, delta)], [1e-8, 5e-4], ls = '--',  color='grey')
axs.plot([gamma_to_kev(max(gmax_star_arr), B, delta), gamma_to_kev(max(gmax_star_arr), B, delta)], [1e-8, 5e-4], ls = '--', color='grey')
axs.plot([gamma_to_kev(gmin, B, delta), gamma_to_kev(gmin, B, delta)], [1e-8, 5e-4], label='Eb', color='r')
axs.legend(fontsize = 14,  loc ='lower right')
 



# axs.legend()
