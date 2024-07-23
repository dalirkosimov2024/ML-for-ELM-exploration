# imports 
import os
import numpy as np
import shutil
import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
from pathlib2 import Path

# function that replaces the .in file variables
def replacer(filename, new_nn):
	file = Path(filename)
	filedata = file.read_text()

	# select the variables to change
	nn_target = "nn="
	nmlow_target = "nmlow="
	nmvac_target = "nmvac="
	nmwinhalf_target = "nmwinhalf="

	# isolate the vairbales
	res1 = filedata.split(nn_target, 1)
	res2 = filedata.split(nmlow_target, 1)
	res3 = filedata.split(nmvac_target,1)
	res4 = filedata.split(nmwinhalf_target, 1)

	# identify the values of the variables
	nn = int("{}{}".format(res1[1][0], res1[1][1]))
	nmlow = int("{}{}".format(res2[1][0], res2[1][1]))
	nmvac = int("{}{}".format(res3[1][0], res3[1][1]))
	nmwinhalf = int("{}{}".format(res4[1][0], res4[1][1]))

	# replace
	newdata1 = filedata.replace(f"nn={nn:02d}", f"nn={new_nn:02d}")
	file.write_text(newdata1)

	file = Path(filename)
	filedata = file.read_text()
	newdata2 = filedata.replace(f"nmlow={nmlow:02d}", f"nmlow=05")
	file.write_text(newdata2)

	file = Path(filename)
	filedata = file.read_text()
	newdata3 = filedata.replace(f"nmvac={nmvac}", f"nmvac={new_nn:002d}")
	file.write_text(newdata3)

	file = Path(filename)
	filedata = file.read_text()
	newdata4 = filedata.replace(f"nmwinhalf={nmwinhalf}", f"nmwinhalf=-1")
	file.write_text(newdata4)

# function that creates new directories for mode numbers and runs ELITE
def driver():

	# initial selections
	name = input("Testcase name: ")
	eqin_condition = input("Select file type (eqdsk/eqin/dskbal): ")

	parent_path = "/home/userfs/l/lcv510/pedestal/results/outputs"
	base_path = f"/home/userfs/l/lcv510/pedestal/results/outputs/{name}"
	os.chdir(parent_path)
	
	# create / go to directory
	if os.path.exists(base_path):
		os.chdir(base_path)
	else:
		os.mkdir(name)
		os.chdir(base_path)

	# commands to run elite
	eq = f"~/pedestal/ELITE/ELITE16/equil/eliteeq -r {name}  "
	vac = f"~/pedestal/ELITE/ELITE16/vac/elitevac -r {name}"
	elite = f"~/pedestal/ELITE/ELITE16/nonsymplas/elite -r {name} -I 1"

	# list of mode numbers
	nn_list = [25,30]

	for nn in nn_list:
		
		os.chdir(base_path)
		
		dir_name = f"nn_{nn}"
		os.mkdir(dir_name)
		new_dir = os.path.join(base_path, dir_name)

		# copy .in file from tescases folder
		shutil.copy(f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/{name}/{name}.in",new_dir)
		
		if eqin_condition == "eqdsk":
			shutil.copy(f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/{name}/{name}.eqdsk",new_dir)
		elif eqin_condition == "eqin":
			shutil.copy(f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/{name}/{name}.eqin",new_dir)
		elif eqin_condition == "dskbal":
			shutil.copy(f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/{name}/{name}.dskbal",new_dir)

		
		os.chdir(new_dir)

		# change the variables
		line_array  = np.array([])
		for filename in os.listdir(new_dir):
			if filename.endswith(".in"):
				replacer(filename, nn)

		# run elite
		print("\n\n\n ----- eq starting ----- \n\n\n")
		os.system(eq)
		print("\n\n\n ----- eq finished ----- \n\n\n")

		print("\n\n\n ----- vac starting ----- \n\n\n")
		os.system(vac)
		print("\n\n\n ----- vac finished ----- \n\n\n")


		print("\n\n\n ----- elite starting ----- \n\n\n")
		os.system("export OMP_NUM_THREADS=32")
		os.system(elite)
		print("\n\n\n ----- elite finished ----- \n\n\n")

# function that reads the growth rates and outputs a nice graph
def extractor(name):
	
	mode_number = np.array([])
	growth_rate = np.array([])
	
	base_path = f"/home/userfs/l/lcv510/pedestal/results/outputs/{name}"

	# Enter the subdirectories
	for folder, subfolder, files in os.walk(base_path):
		for file in files:
			
			# Look for the fole containing the 
			# growth rates, in ".gamma" files
			if file.endswith(".gamma"):
				file_path = os.path.join(folder,file)

				# Open the ".gamma" file, convert the ASCII
				#    contents into an array and close the file
				g = open(file_path, "r")
				line_array = np.array([])
				for line in g:
					line = line.strip()
					columns = np.array(line.split())
					line_array = np.append(line_array,columns)
				g.close()

				# Print values
				print(f"Mode number: {line_array[9]}")
				print(f"Mode number: {line_array[10]}")

				#  Append desired values from the line array into
				#   our initial empty array
				mode_number = np.append(mode_number, float(line_array[9]))
				growth_rate = np.append(growth_rate, float(line_array[10]))

	print(mode_number)
	plt.plot(sorted(mode_number), growth_rate[np.argsort(mode_number)], ".")
	plt.plot(sorted(mode_number), growth_rate[np.argsort(mode_number)], "--", alpha=0.4)
	plt.xlabel("Toroidal Mode Number, n") ; plt.ylabel("Growth Rate " r"($\gamma$)")
	plt.title(f"Growth Rate for a given Mode number for the {name} testcase")
	plt.savefig("mu45272_nn.png")
	max_growth_rate = np.max(growth_rate)
	print(f"Max growth rate: {max_growth_rate}")
	return max_growth_rate

if __name__ == "__main__":

	action = input("Drive or Extract? d/e ")
	if action == "d":
		driver()
	elif action=="e":
		name = input("Testcase name: ")
		extractor(name)


		

		



		

