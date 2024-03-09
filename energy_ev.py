# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:07:42 2024

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
from scipy.optimize import curve_fit
import scipy
from scipy.odr import ODR, Model, Data, RealData

import scienceplots

plt.style.use(['science', 'ieee'])

plt.close('all')

theta = np.pi/2 #degrees
sintheta = np.sin(theta)

k = 1
p = 4
B = 1 #Gauss

sigma_T = 6.65e-25
c = 3.00e+10
u_B = B*B/(8.0*np.pi)
m_e = 9.11e-28
h = 6.6261e-27

A = (4/3 * sigma_T * u_B)/(m_e * c)

e = 4.8e-10
gmin = 10
gmax = 1e6
g0 = 800
erg_to_kev = 6.242e+8
delta = 2*300

B0 = B
td = 80
m = 2

nu_min = (e*B)/(2*np.pi*m_e*c) * gmin**2
nu_max = (e*B)/(2*np.pi*m_e*c) * gmax**2

t_dat = [1, 3, 5, 7, 9, 11, 14, 17, 19, 21, 23, 25, 28, 32, 36, 40, 44, 48, 55, 65, 85, 95, 105, 115, 125, 135, 150, 170]
t_err = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10]
Ep_dat = [2411.3, 951.6, 721, 1543, 1053.6, 567.1, 257.3, 275, 580, 319.6, 335.2, 305.5, 83.9, 69.8, 225.6, 76.3, 37.7, 42.2, 35.5, 36.4, 7.74, 5.81, 4.59, 4.88, 3.30, 4.04, 3.31, 2.51]
Ep_err = [602, 47.4, 47.5, 47.5, 72, 135.7, 205.7, 40.9, 86.5, 45.8, 51.5, 47.9, 30.3, 16.9, 50.9, 15.9, 3.06, 5.61, 2.29, 4.32, 0.81, 0.83, 0.54, 0.89, 0.37, 0.67, 0.49, 0.30]
Eb_dat = [21, 46.7, 30.7, 64.3, 30, 28.5, 19.9, 20.6, 27.2, 23.8, 24.2, 24.8, 24.3, 17.2, 20.2, 17.6]
Eb_err = [8.8, 8.2, 3.3, 4.9, 2.7, 2.4, 3.6, 4.2, 2.8, 2.9, 2.6, 2.7, 0.9, 3.7, 1.6, 4.2]

def t_syn(gamma, B):
    u_B = B*B/(8.0*np.pi)
    return (gamma*m_e*c)/(4/3 * sigma_T * u_B * gamma**2)

def gamma_0(t):
    return (m_e*c)/(4/3 * sigma_T * u_B * t)

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
    
def Fg(x, g):
    
    res = []
    
    for gam in g:
        res.append(x/nu_c(gam) * quad(Bessel5_3, x/nu_c(gam), np.inf)[0])
    
    return np.array(res)

def B_mag(t):
    return B0*(t/td)**(-m)

def E_func(t_arr, B, gmax, delt):
    t_rf = t_arr*delt
    ub = B*B/(8.0*np.pi)
    a = (4/3 * sigma_T * ub)/(m_e * c)
    
    gstar = gmax/(1+a*t_rf*gmax)
    nu_s = e*B * gstar**2 / (2*np.pi*m_e*c)
    E_rf = nu_s * h
    E_of = delt*E_rf
    
    return E_of*erg_to_kev

def func(beta, t_arr):
    t_rf = t_arr*beta[2]
    ub = beta[0]*beta[0]/(8.0*np.pi)
    a = (4/3 * sigma_T * ub)/(m_e * c)
    
    gstar = beta[1]/(1+a*t_rf*beta[1])
    nu_s = e*beta[0] * gstar**2 / (2*np.pi*m_e*c)
    E_rf = nu_s * h
    E_of = beta[2]*E_rf
    
    return E_of*erg_to_kev

def gmin_func(t_arr, B, gmin, delt):
    ub = B*B/(8.0*np.pi)
    a = (4/3 * sigma_T * ub)/(m_e * c)
    
    nu_s = e*B * gmin**2 / (2*np.pi*m_e*c)
    E_rf = nu_s * h
    E_of = delt*E_rf
    E_of *= erg_to_kev
    return np.array([E_of for t in t_arr])

nu_nuc = np.logspace(-3, 1, 100)
nu = np.logspace(np.log10(nu_min)-2, np.log10(nu_max)+2, 100)
gamma = np.logspace(-1, 4, 100)
N = np.zeros(len(gamma))
t = np.logspace(0, 3, 100)
nu_syn = np.zeros(len(t))
E_obs = np.zeros(len(t))
E_rest = np.zeros(len(t))

fig, axs = plt.subplots(num = 'kev energy')
    
#data = RealData(t_dat, Ep_dat, t_err, Ep_err) uncomment this line to use all of the data
data = RealData(t_dat[0:16], Ep_dat[0:16], t_err[0:16], Ep_err[0:16])

#ODR fit
model = Model(func)
odr = ODR(data, model, [1, 1e5, 300])
output = odr.run()

axs.plot(t, func(output.beta, t), label='ODR fit')

#lsqrs fit for gmax*

#popt, pconv = curve_fit(E_func, t_dat, Ep_dat, p0=[1, 1e5, 300], sigma=Ep_err, bounds=((1e-5, 1e3, 200),(1e2, 1e7, 1000)))
popt, pconv = curve_fit(E_func, t_dat[0:16], Ep_dat[0:16], p0=[1, 1e5, 300], sigma=Ep_err[0:16], bounds=((1e-5, 1e3, 100),(1e3, 1e7, 2000)))

#fitting gamma_min using ODR fit's params
poptgmin, pconvgmin = curve_fit(lambda x, A: gmin_func(x, output.beta[0], A, output.beta[2]), t_dat[0:16], Eb_dat, p0=100, sigma=Eb_err, bounds=(10, 1e6))

#axs.plot(t, E_func(t, *popt), label ='Least squares fit')
axs.plot(t, gmin_func(t, output.beta[0], poptgmin[0], output.beta[2]), label='Least squares fit')
axs.plot(t, func([output.beta[0], poptgmin[0], output.beta[2]], t), label='$\gamma_{min}$ cooling fit')
axs.plot([t_syn(poptgmin[0], popt[0])/popt[2], t_syn(poptgmin[0], popt[0])/popt[2]], [0, 1e4], color = 'black', linestyle = 'dashed')
axs.errorbar(t_dat[0:20], Ep_dat[0:20], xerr=t_err[0:20], yerr=Ep_err[0:20], markersize=2, marker=".", ls='none', label='Ep', c='blue')
axs.errorbar(t_dat[21:28], Ep_dat[21:28], xerr=t_err[21:28], yerr=Ep_err[21:28], markersize=2, marker=".", ls='none', color='gray')
axs.errorbar(t_dat[0:16], Eb_dat, xerr=t_err[0:16], yerr = Eb_err, markersize=2, marker=".", ls='none', label='Eb', color='green')
nu_syn = np.zeros(len(t))
E_obs = np.zeros(len(t))
E_rest = np.zeros(len(t))
    
        
axs.set_xlabel('Observer Frame Time [seconds]')
axs.set_ylabel('Observer Frame energy [keV]')
axs.set_xscale('log')
axs.set_yscale('log')
axs.legend()


### time dependant mag field. Needs more work...

nu_syn = np.zeros(len(t))
E_obs = np.zeros(len(t))
E_rest = np.zeros(len(t))

fig, axs = plt.subplots(num = 'kev energy time')

#for ti in range(10, 100, 20):
#    td = ti
for i in range(len(t)):
    u_B = B_mag(t[i])**2 / (8.0*np.pi)
    A = (4/3 * sigma_T * u_B)/(m_e * c)
    gamma_star = gmax/(1+A*gmax*t[i])
    #gamma_star = gmin
    
    nu_syn[i] = e*B_mag(t[i]) * gamma_star**2 / (2*np.pi*m_e*c)
    E_rest[i] = nu_syn[i]*h
    E_obs[i] = delta*E_rest[i]

axs.scatter(t/delta, 50*E_obs*erg_to_kev, s=2, label = 't_d = {} seconds'.format(td))    
nu_syn = np.zeros(len(t))
E_obs = np.zeros(len(t))
E_rest = np.zeros(len(t))
    
axs.set_xlabel('Obs Time (seconds)')
axs.set_ylabel('Observer Frame energy (kev)')
axs.scatter(t_dat, Ep_dat, s=2, label='Article data')
axs.set_xscale('log')
axs.set_yscale('log')
axs.legend()






