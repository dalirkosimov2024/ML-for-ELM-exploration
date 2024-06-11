import os
import numpy as np
import shutil
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("Agg")



def driver():

	name = input("Testcase name: ")
	init_nn = input("Initial mode number: ")
	eqin_condition = input("Does testcase have an initial .eqin file? (y/n): ")

	parent_path = "/home/userfs/l/lcv510/pedestal/results/outputs"
	os.chdir(parent_path)
	os.mkdir(name)

	eq = f"~/pedestal/ELITE/ELITE16/equil/eliteeq -r {name} "
	vac = f"~/pedestal/ELITE/ELITE16/vac/elitevac -r {name}"
	elite = f"~/pedestal/ELITE/ELITE16/symplas/elite -r {name} -I 0"
	base_path = f"/home/userfs/l/lcv510/pedestal/results/outputs/{name}"

	nn_list = np.array([])
	for x in range(5,25, 5):
		nn_list = np.append(nn_list,int(x))
	print(nn_list,"\n")


	os.chdir(base_path)
	print(os.getcwd(),"\n")


	for nn in nn_list:
		
		os.chdir(base_path)
		
		dir_name = f"nn_{nn}"
		os.mkdir(dir_name)
		new_dir = os.path.join(base_path, dir_name)
		
		shutil.copy(f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/{name}/{name}.in",new_dir)
		
		if eqin_condition == "y":
			shutil.copy(f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/{name}/{name}.eqdsk",new_dir)
		else:
			shutil.copy(f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/{name}/{name}.dskbal",new_dir)
		
		os.chdir(new_dir)

		line_array  = np.array([])
		for filename in os.listdir(new_dir):
			if filename.endswith(".in"):
				g = open(filename, "r")
				filedata = g.read()
				g.close()

				newdata = filedata.replace(f"n= {init_nn}", f"n= {nn}")

				g = open(filename, "w")
				g.write(newdata)
				g.close()
		os.system(eq)
		os.system(vac)
		os.system(elite)


def extractor(name):
	# 0) Name of testcase
	
	# 1) Make empty array list to append to later
	mode_number = np.array([])
	growth_rate = np.array([])
	
	base_path = f"/home/userfs/l/lcv510/pedestal/results/outputs/{name}"

	# 3) Enter the subdirectories
	for folder, subfolder, files in os.walk(base_path):
		for file in files:

			# 4) Look for the fole containing the 
			#    growth rates, in ".gamma" files
			if file.endswith(".gamma"):
				file_path = os.path.join(folder,file)

				# 5) Open the ".gamma" file, convert the ASCII
				#    contents into an array and close the file
				g = open(file_path, "r")
				line_array = np.array([])
				for line in g:
					line = line.strip()
					columns = np.array(line.split())
					line_array = np.append(line_array,columns)
				g.close()

				# 6) Print values
				print(f"Mode number: {line_array[9]}")
				print(f"Mode number: {line_array[10]}")

				# 7) Append desired values from the line array into
				#    our initial empty array
				mode_number = np.append(mode_number, float(line_array[9]))
				growth_rate = np.append(growth_rate, float(line_array[10]))

	#plt.plot(mode_number, growth_rate, ".")
	#plt.plot(mode_number, growth_rate, "--", alpha=0.4)
	#plt.xlabel("Toroidal Mode Number, n") ; plt.ylabel("Growth Rate " r"($\gamma$)")
	#plt.title(f"Growth Rate for a given Mode number for the {name} testcase")
	#plt.show()
	max_growth_rate = np.max(growth_rate)
	print(f"Max growth rate: {max_growth_rate}")
	return max_growth_rate

def max_growth_rate_calc():
	max_growth_rate_array = np.array([])
	name_array = np.array([])
	base_path = "/home/userfs/l/lcv510/pedestal/results/outputs"
	for folder in os.scandir(base_path):
		name = input("Tescase name: ")
		gamma  = extractor(name)
		max_growth_rate_array = np.append(max_growth_rate_array, gamma)
		name_array = np.append(name_array, name)
	print(max_growth_rate_array)
	print(name_array)



if __name__ == "__main__":

	action = input("Drive or Extract? d/e ")
	if action == "d":
		driver()
	elif action=="e":
		name = input("Testcase name: ")
		extractor(name)
	else:
		max_growth_rate_calc()

		

		



		


