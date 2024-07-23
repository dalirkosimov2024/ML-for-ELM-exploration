#imports
from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# function that allows for the animation to run
def update(frame):
    scat.set_offsets(sample[:frame+1])
    scat.set_facecolor("red")
    return scat,

# latin hypercube sampler
n = 16
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=n)
 
# plotting 
fig, ax = plt.subplots()
scat = ax.scatter([],[], marker="x", s=80)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
#plt.scatter(sample[:,0],sample[:,1],c='r') 

# plotting the grids for better visualisation
for i in np.arange(0,1,1/n):
    plt.axvline(i)
    plt.axhline(i)
plt.axhline(y=0.5, color="black", linewidth=2)
plt.axvline(x=0.5, color="black", linewidth=2)
plt.suptitle(f"Latin Hypercube demo with {n} points")

# animation runner in matplotlib
ani = FuncAnimation(fig, update, frames=len(sample),
                     interval=300, blit=True, repeat = False)
plt.show()
