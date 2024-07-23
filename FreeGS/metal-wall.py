#!/usr/bin/env python
#
# Calculate the equilibrium for a plasma surrounded by a metal wall
# This is done by creating a ring of coils, with feedback control setting
# the poloidal flux to zero at the location of each coil.

#imports
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

# main function - runs FreeGS and outputs a geqdsk file
def main(resolution):

	# Miller parameters
	R0 = 1.0 # Middle of the circle
	rwall = 0.5  # Radius of the circular wall
	b = 0 # indentation
	delta = 0.3 # triangularity
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

	wall = Wall([0.7, 1.1, 1.4, 1.6, 1.4, 1.1, 0.7, 0.5, 0.4, 0.5],
		    [0.8, 0.8, 0.5, 0, -0.5, -0.8, -0.8, -0.5, 0 , 0.5])

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

	# pprime function that works
	def pprime_func(r, a0=0.98, a1=0.01, a2 = 110000, a3=0.1, a4=0.06):
		x = (a0 - r) / (2*a1)
		return (a2 - a4) / 2 * (exp(2*x)*(a3*exp(2*x)+2*a3*x+a3+4)) / (exp(2*x)+1)**2 

	# another pprime function (doesnt work)
	def pprime_func_long(r):	
		p = (exp(a0/a1)*((a3*r-(a1+a0)*a3-4*a1)*exp(r/a1)-a1*exp(a0/a1)*a3)) / ( 4*a1**3*(exp(r/a1)+exp(a0/a1))**2 )
		return p
		
	# ffprime function that mimics MASTU (doesnt work)
	def ffprime_func_realistic(r,a0 = 0.98, a1=0.01, a2=0.01, a3=0.08, a4=-0.5, b=a3, c=0):
		x = (a0 - r) / (2 * a1)
		mtanh = ( (1 + b*x)*exp(x) - (1+c*x)*exp(-x) ) / ( exp(x) + exp(-x) )
		return -(a2 - a4) / 2 * ( mtanh + 1 ) + a4 +2
		
	# ffprime function that works
	def ffprime_func(x):
		return exp(-6*x+1.8)
		
	# custom profile with pprime and ffprime
	custom_profile = freegs.jtor.ProfilesPprimeFfprime(pprime_func=pprime_func, ffprime_func=ffprime_func, fvac = 0.2)

	#########################################
	# Coil current constraints

	# Same location as the coils
	psivals = [ (R, Z, 0.0) for R, Z in zip(Rwalls, Zwalls) ]

	constrain = freegs.control.constrain(psivals=psivals)

	#########################################
	# Nonlinear solve

	freegs.solve(eq,          # The equilibrium to adjust
		     custom_profile,    # The toroidal current profile function
		     constrain,   # Constraint function to set coil currents
		     psi_bndry=0.0,
		     show=False,   # Because no X-points, specify the separatrix psi
		     maxits=100)  

	#eq now contains the solution

	print("Done!")

	print("Plasma current: %e Amps" % (eq.plasmaCurrent()))
	print("Plasma pressure on axis: %e Pascals" % (eq.pressure(0.0)))
	print("Plasma pressure on 0.3: %e Pascals" % (eq.pressure(0.3)))
	print("Plasma pressure on 0.5: %e Pascals" % (eq.pressure(0.5)))
	print("Plasma pressure on 0.9: %e Pascals" % (eq.pressure(0.9)))

	##############################################
	# Save to G-EQDSK file

	from freegs import geqdsk

	path = "/home/userfs/l/lcv510/pedestal/ELITE/testcases/metal-wall/"

	with open(os.path.join(path, "metal-wall.geqdsk"), "w") as f:
	    geqdsk.write(eq, f)

	shutil.copyfile(os.path.join(path, "metal-wall.geqdsk"),os.path.join(path, "metal-wall.eqdsk"))

	##############################################
	# Final plot

	axis = eq.plot(show=False)
	constrain.plot(axis=axis, show=False)

# reader function that reads geqdsk files
def geqdsk_reader(filename):

	path = f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/{filename}/{filename}.geqdsk"

	with open(path, "r") as f:
	    data = geqdsk.read(f)

	# get value of what we want
	pressure = data["pres"]
	ffprime = data["ffprime"]
	pprime = data["pprime"]
	q = data["qpsi"]

	# plot
	plt.plot(np.linspace(0, 1, len(ffprime)), ffprime)
	plt.suptitle("ffprime against flux")
	plt.xlabel(r"Normalised flux ($\psi$)")
	plt.ylabel("ffprime")
	plt.show()

	plt.plot(np.linspace(0, 1, len(pprime)), pprime)
	plt.title("pprime against flux")
	plt.xlabel(r"Normalised flux ($\psi$)")
	plt.ylabel("pprime")
	plt.show()
	print(pprime)

	plt.plot(np.linspace(0, 1, len(pressure)), pressure)
	plt.title("pressure against flux")
	plt.xlabel(r"Normalised flux ($\psi$)")
	plt.ylabel("pressure")
	plt.show()
	print(pressure)

	plt.plot(np.linspace(0,1,len(q)), q)
	plt.title("Safety factor against flux")
	plt.xlabel(r"Normalised flux ($\psi$)")
	plt.ylabel("Safety factor, q")
	plt.show()
	
if __name__ == "__main__":
	x = input("Drive or read? (d/r):") 
	if x == "r":
		filename = input("Filename: ")
		geqdsk_reader(filename)
	else:
		resolution = int(input("Resolution: (65/257/513) "))
		main(resolution)
