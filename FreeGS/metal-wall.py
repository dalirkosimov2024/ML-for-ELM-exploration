
#!/usr/bin/env python
#
# Calculate the equilibrium for a plasma surrounded by a metal wall
# This is done by creating a ring of coils, with feedback control setting
# the poloidal flux to zero at the location of each coil.

import freegs
import numpy as np
from numpy import exp
import time
from freegs.machine import Wall
from freeqdsk import geqdsk
import matplotlib
import matplotlib.pyplot as plt
import shutil
import os
import sys

#########################################
# Create a circular metal wall by using a ring of coils and psi constraints

def main(name, delta, resolution):


	R0 = 1.0 # Middle of the circle
	rwall = 0.5  # Radius of the circular wall
	b = 0 # indentation
	delta = float(delta) # triangularity
	kappa = 1.5 # elongation

	npoints = 200 # Number of points on the wall

	# Poloidal angles
	thetas = np.linspace(0, 2*np.pi, npoints, endpoint=False)

	# Points on the wall
	Rwalls = R0 -b + (rwall + b * np.cos(thetas)) * np.cos(thetas + delta * np.sin(thetas))
	Zwalls = kappa * rwall * np.sin(thetas)

	#########################################
	# Create the machine, which specifies coil locations
	# and equilibrium, specifying the domain to solve over

	coils = [ ("wall_"+str(theta), freegs.machine.Coil(R, Z))
		  for theta, R, Z in zip(thetas, Rwalls, Zwalls) ]

	wall = Wall([0.3, 1.7, 1.7, 0.3],
                    [0.85, 0.85, -0.85, -0.85])

	tokamak = freegs.machine.Machine(coils, wall)

	eq = freegs.Equilibrium(tokamak=tokamak,
				Rmin=0.1, Rmax=2.0,    # Radial domain
				Zmin=-1.0, Zmax=1.0,   # Height range
				nx=resolution, ny=resolution,          # Number of grid points
				boundary=freegs.boundary.freeBoundaryHagenow)  # Boundary condition

	#########################################
	# Plasma profiles

	profiles = freegs.jtor.ConstrainPaxisIp(eq,
						1e5, # Plasma pressure on axis [Pascals]
						1e6, # Plasma current [Amps]
						2.0) # Vacuum f=R*Bt

	# DALIR -- pprime function variables

	a0 = 0.96
	a1 = 0.004
	a2 = 1000
	a3 = 0.05
	a4 = 0.06
	c = 0.01


	# DALIR -- pprime function
	def pprime_func(r):
		x = (a0 - r) / (2*a1)
		return (a2 - a4) / 2 * (exp(2*x)*(a3*exp(2*x)+2*a3*x+a3+4)) / (exp(2*x)+1)**2 

	def pprime_func2(r):
		return (a2-a4)/2*(exp(a0/a1)*((b*r+(-a1-a0)*b-4*a1)*exp(r/a1)-a1*exp(a0/a1)*b)) / (2*a1**2*(exp(r/a1)+exp(a0/a1))**2)

		
	def ffprime_func2(r,a0 = 0.96, a1=0.01, a2=0.01, a3=0.08, a4=-0.5, b=a3, c=0.5):
		x = (a0 - r) / (2 * a1)
		mtanh = ( (1 + b*x)*exp(x) - (1+c*x)*exp(-x) ) / ( exp(x) + exp(-x) )
		return -(a2 - a4) / 2 * ( mtanh + 1 ) + a4 +0.8

	def ffprime_func(x):
		return 6*exp(-5*x)



   

	# DALIR -- custom profile with pprime and ffprime
	custom_profile = freegs.jtor.ProfilesPprimeFfprime(pprime_func=pprime_func, ffprime_func=ffprime_func, fvac = 0.2)

	#########################################
	# Coil current constraints
	#

	# Same location as the coils
	psivals = [ (R, Z, 0.0) for R, Z in zip(Rwalls, Zwalls) ]

	constrain = freegs.control.constrain(psivals=psivals)

	#########################################
	# Nonlinear solve

	freegs.solve(eq,          # The equilibrium to adjust
		     custom_profile,    # The toroidal current profile function
		     constrain,   # Constraint function to set coil currents
		     psi_bndry=0.0,
		     show=True,
		     maxits=150)  # Because no X-points, specify the separatrix psi


	#eq now contains the solution

	print("Done!")

	print("Plasma current: %e Amps" % (eq.plasmaCurrent()))
	print("Plasma pressure on axis: %e Pascals" % (eq.pressure(0.0)))
	print("Plasma pressure on 0.5: %e Pascals" % (eq.pressure(0.5)))
	print("Plasma pressure on 0.9: %e Pascals" % (eq.pressure(0.9)))


	##############################################
	# Save to G-EQDSK file

	from freegs import geqdsk

	os.chdir("/home/userfs/l/lcv510/pedestal/ELITE/testcases/metal-wall/triangularity_sweep")
	os.mkdir(f"{name}")
	os.chdir(f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/metal-wall/triangularity_sweep/{name}")

	path = f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/metal-wall/triangularity_sweep/{name}/{name}.geqdsk"

	with open(path, "w") as f:
	    geqdsk.write(eq, f)

	shutil.copyfile(path, f"{name}.eqdsk")

	new_path = f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/metal-wall/triangularity_sweep/{name}/{name}.eqdsk"

	with open(new_path, "r") as f:
		lines = f.readlines()

	if len(lines) > 0 : 
		lines[0] = lines[0][51:]
		lines[0] = " " * 3 +lines[0]

	with open(new_path, "w") as f:
		f.writelines(lines)

	##############################################
	# Final plot

	axis = eq.plot(show=False)
	constrain.plot(axis=axis, show=False)

	import matplotlib.pyplot as plt
	"""
	#plt.plot(*eq.q())
	#plt.xlabel(r"Normalised $\psi$")
	#plt.ylabel("Safety factor")
	#plt.suptitle(r"Safety Factor against Polodial Flux ($\psi$)")
	#plt.show()

	#plt.plot(eq.pressure())
	#plt.suptitle("pressure")
	#plt.show()

	# plot the pressure
	psi = np.linspace(0, 1)
	pressure = eq.pressure(psi)
	plt.plot(psi, pressure)
	plt.xlabel(r"Normalised $\psi$")
	plt.ylabel("Pressure (Pa)")
	plt.suptitle(r"Pressure against Polodial Flux ($\psi$)")
	plt.show()
	"""





def geqdsk_reader(name):

	path = f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/metal-wall/triangularity_sweep/{name}/{name}.geqdsk"

	with open(path, "r") as f:
	    data = geqdsk.read(f)

	pressure = data["pres"]
	ffprime = data["ffprime"]
	pprime = data["pprime"]
	q = data["qpsi"]
	fpol = data["fpol"]

	# plot
	plt.plot(np.linspace(0, 1, len(ffprime)), ffprime)
	plt.suptitle("ffprime against flux")
	plt.xlabel(r"Normalised flux ($\psi$)")
	plt.ylabel("ffprime")
	plt.show()

	plt.plot(np.linspace(0, 1, len(fpol)), fpol)
	plt.suptitle("fpol against flux")
	plt.xlabel(r"Normalised flux ($\psi$)")
	plt.ylabel("fpol")
	plt.show()


	plt.plot(np.linspace(0, 1, len(pprime)), pprime)
	plt.title("pprime against flux")
	plt.xlabel(r"Normalised flux ($\psi$)")
	plt.ylabel("pprime")
	plt.show()

if __name__ == "__main__":

	delta = sys.argv[1]
	name = f"delta_{delta}"
	resolution = 65
	main(name, delta, resolution)

