# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:28:43 2024

@author: berej
"""

import numpy as np
from matplotlib import pyplot as plt, ticker as mticker
import matplotlib as mpl
import scipy.special as sci
from scipy.integrate import quad
from scipy.integrate import fixed_quad
from scipy.integrate import trapezoid
from scipy.integrate import romberg
from scipy.integrate import simpson
import scipy
import scienceplots
from matplotlib.ticker import AutoMinorLocator

plt.style.use(['science', 'ieee'])

plt.close('all')


theta = np.pi/2 #degrees
sintheta = np.sin(theta)

k = 1
p = 2
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
#t_array = np.array([1, 2, 6, 8, 10, 12, 18, 20, 30, 34, 50, 60, 80, 100, 120])
#t_array = np.linspace(50, 60, 10)

fig, axs = plt.subplots(2, 3, num='t0-2', figsize=(15, 15), constrained_layout=True, sharey=True)
fig.subplots_adjust(wspace=0)
fig.subplots_adjust(hspace=0.7)

figg, axsg = plt.subplots(2, 3, num='gammas', figsize=(15, 15), constrained_layout=True, sharey=True)
figg.subplots_adjust(wspace=0)
figg.subplots_adjust(hspace=0.7)

#n_lines = 10
#cl = np.arange(start, stop, step)
cl = np.arange(0, 130, 10)
#clb = np.arange(start, stop, step*2)
#clb = cl
norm = mpl.colors.Normalize(vmin=cl.min(), vmax=cl.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
cmap.set_array([])

for q in range(len(timeframes)):
    J_array = []
    N_array = []
    t_array = np.linspace(timeframes[q][0], timeframes[q][1], 10)

    for t in t_array:
        t *= delta
        
        for i in range(len(gamma)):
            
            if gmax/(1+A*gmax*t) >= gmin:
                if gmax/(1+A*gmax*t) >= gamma[i] and gamma[i] >= gmin:
                    N[i] = q0s[q]*gamma[i]**(-(p+1)) / (A*(p-1)) * (1 - (1 - A*gamma[i]*t)**(p-1))
                if gmax/(1+A*gmax*t) < gamma[i] and gamma[i] <= gmax:
                    N[i] = q0s[q]*gamma[i]**(-2) / (A*(p-1)) * (gamma[i]**(-p+1) - gmax**(-p+1))
            else:
                if gamma[i] >= gmax/(1+A*gmax*t) and gamma[i] < gmin:
                    N[i] = q0s[q]*(gamma[i])**(-2)/(A*gmin*(p-1))
                if gamma[i] >= gmin and gamma[i] <= gmax:
                    N[i] = q0s[q]*gamma[i]**(-2) / (A*(p-1)) * (gamma[i]**(-p+1) - gmax**(-p+1))
        t *= 1/delta

        for n in nu:
            print('at nu {nu_text:2.2e}'.format(nu_text=n))
            J_syn.append(np.trapz(N*Fg(n, gamma), gamma))

        J_array.append(delta**4*nu*np.array(J_syn)/(4*np.pi*D**2))
        N_array.append(N)
        N = np.zeros(len(gamma))
        J_syn = []

    N_arr = np.array(N_array)
    J_arr = np.array(J_array)
    gmax_star_arr = gstar(gmax, t_array)
    gmax_star_mean = np.average(gmax_star_arr)
    
    #uncomment if you wanna save data. I recommend implementing a way of using previously saved data to plot.
    #np.savetxt('./results/res{}-{}s.txt'.format(timeframes[q][0], timeframes[q][1]), np.array([delta*nu*h*erg_to_kev, np.average(J_arr, axis=0)]))
    #axsJ.legend()
    axsg[q//3][q-(q//3)*3].plot(gamma, np.average(N_arr, axis=0))
    axsg[q//3][q-(q//3)*3].set_xscale('log')
    axsg[q//3][q-(q//3)*3].set_yscale('log')
    axsg[q//3][q-(q//3)*3].set_xlabel(r'$\log{\gamma}$', fontsize=5)
    
    art_dat = np.loadtxt('./data/{}-{}s.txt'.format(timeframes[q][0], timeframes[q][1]), unpack=True, skiprows=1)
    axs[q//3][q-(q//3)*3].errorbar(art_dat[0], art_dat[2]/erg_to_kev, xerr=art_dat[1], yerr=art_dat[3]/erg_to_kev, markersize=2, marker=".", ls='none', label='Article Data', alpha=0.2, color='green')
    axs[q//3][q-(q//3)*3].plot(delta*nu*h*erg_to_kev, np.average(J_arr, axis=0), label='{}-{}s average'.format(timeframes[q][0], timeframes[q][1]), color='black', linestyle='solid')
    axs[q//3][q-(q//3)*3].plot([gamma_to_kev(gmin), gamma_to_kev(gmin)], [1e-8, 1e-4], label='Eb', color='blue', linestyle = 'dashed')
    axs[q//3][q-(q//3)*3].plot([gamma_to_kev(gmax_star_mean), gamma_to_kev(gmax_star_mean)], [1e-8, 1e-4], label='Ep', color='red', linestyle = 'dashed')
    axs[q//3][q-(q//3)*3].set_title('{} - {}s'.format(timeframes[q][0], timeframes[q][1]), fontsize=7, loc='right')
    axs[q//3][q-(q//3)*3].set_xscale('log')
    axs[q//3][q-(q//3)*3].set_yscale('log')
    axs[q//3][q-(q//3)*3].set_xlim((1e1, 1e4))
    axs[q//3][q-(q//3)*3].set_ylim((1e-8, 1e-4))
    axs[q//3][q-(q//3)*3].set_xticks((np.arange(1e1, 1e4, 10)), fontsize=5)
    
    axs[q//3][q-(q//3)*3].set_xlabel(r'Observer Frame Energy $[keV]$', fontsize=5)
    axs[q//3][q-(q//3)*3].set_xticks([1e1, 100, 1000])
    axs[q//3][q-(q//3)*3].tick_params(axis='both', which='major', labelsize=5)
    axs[q//3][q-(q//3)*3].xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    axs[q//3][q-(q//3)*3].yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    axs[q//3][q-(q//3)*3].yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    axs[q//3][q-(q//3)*3].legend(fontsize=3.5, loc='best', frameon=True)

axs[0][0].set_ylabel(r'$\log{\nu F_{\nu}}$ $[erg/cm^{2}/s]$', fontsize=7)
axs[1][0].set_ylabel(r'$\log{\nu F_{\nu}}$ $[erg/cm^{2}/s]$', fontsize=7)

axsg[0][0].set_ylabel(r'$\log{N(\gamma)}$', fontsize=7)
axsg[1][0].set_ylabel(r'$\log{N(\gamma)}$', fontsize=7)

