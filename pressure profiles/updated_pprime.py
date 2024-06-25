#imports 

import numpy as np
from numpy import exp 
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import sympy as sp
from scipy.misc import derivative

# global variables 
a0 = 0.8
a1 = 0.01
a2 = 0.6
a3 = 0.04
a4 = 0.05
b = a3

# MASTU like space, r is minor radius
r = np.linspace(0,1,1000)

# redefine space
x = (a0  - r) / (2 * a1) 

print(x)
# modified tanh function
def mtanh_func(x):
    return ( (1 + b*x)*exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )
mtanh = mtanh_func(x)
  
# this function outputs the H-mode pressure profile pedestal shape
def edge_func(mtanh):
    return (a2 - a4) / 2 * ( mtanh + 1 ) + a4
edge = edge_func(mtanh)

# find derivative of the pressure profile for dp/dr
d_edge = np.gradient(edge) # either do it by hand or use scipy interpolate 

def pprime_func(x):
  return -(a2 - a4) / 2 * (exp(2*x)*(a3*exp(2*x) + 2*a3*x +  a3 + 4)) / (exp(2*x)+1)**2

pprime = pprime_func(x)
pprime_1 = pprime_func(x)
    


plt.plot(r, edge, label = "pressure profile")
plt.plot(r, pprime_1, label="derivative (pprime)") 
plt.axvline(a0, label = "transport barrier", color="red", linestyle="--")
plt.title("Pressure profile and its derivative against poloidal flux") 
plt.xlabel(r"Normalised Flux ($\psi$)")
plt.ylabel("Normalised Pressure (Pa)")
plt.legend(loc=6); plt.show()



