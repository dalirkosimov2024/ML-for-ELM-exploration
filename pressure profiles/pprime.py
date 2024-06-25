#imports 

import numpy as np
from numpy import exp 
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import sympy as sp
from scipy.misc import derivative

# global variables 
a0 = 0.5
a1 = 0.01
a2 = 600
a3 = 0.04
a4 = 0.05
b = a3

# MASTU like space, r is minor radius
r = np.linspace(0,0.6,1000)

# redefine space
x = (a0  - r) / (2 * a1) 

# modified tanh function
def mtanh_func(x,b):
    return ( (1 + b*x)*exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )
mtanh = mtanh_func(x, b)
  
# this function outputs the H-mode pressure profile pedestal shape
def edge_func(mtanh):
    return (a2 - a4) / 2 * ( mtanh + 1 ) + a4
edge = edge_func(mtanh)

# find derivative of the pressure profile for dp/dr
d_edge = np.gradient(edge)

# plotting pressure profile 
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(r, edge) 
ax.axvline(a0, linestyle="--", color = "red")
plt.suptitle('H-mode pedestal formed from a modified tanh function')
ax.set_xlabel('Minor Radius, r (m)')
ax.set_ylabel('Pressure (Pa)')

transport_barrier = mlines.Line2D([], [], color="red", linestyle="--",
                                   label="Transport barrier")
ax.legend(
            handles=[transport_barrier],
            bbox_to_anchor=(1, 1),
            loc="upper left",)
plt.tight_layout()
plt.show()

# plotting dp/dr 
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(r, d_edge) 
ax.axvline(a0, linestyle="--", color = "red")
plt.suptitle('Plot of the derivative of the mtanh function dP/dr')
ax.set_xlabel('Minor Radius, r (m)')
ax.set_ylabel('dP/dr')
transport_barrier = mlines.Line2D([], [], color="red", linestyle="--",
                                   label="Transport barrier")
ax.legend(
            handles=[transport_barrier],
            bbox_to_anchor=(1, 1),
            loc="upper left",)
plt.tight_layout()
plt.show()



