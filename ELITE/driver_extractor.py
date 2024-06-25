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

	parent_path = "/home/userfs/l/lcv510/pedestal/results"
	os.chdir(parent_path)
	os.mkdir(name)

	eq = f"~/pedestal/ELITE/ELITE16/equil/eliteeq -r {name} "
	vac = f"~/pedestal/ELITE/ELITE16/vac/elitevac -r {name}"
	elite = f"~/pedestal/ELITE/ELITE16/symplas/elite -r {name} -I 0"
	base_path = f"/home/userfs/l/lcv510/pedestal/results/{name}"

	nn_list = np.array([])
	for x in range(5,45, 5):
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
			shutil.copy(f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/{name}/{name}.equin",new_dir)
		else:
			shutil.copy(f"/home/userfs/l/lcv510//pedestal/ELITE/testcases/{name}/{name}.dskbal",new_dir)
		
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


def extractor():
	# 0) Name of testcase
	name = input("Testace name: ")

	# 1) Make empty array list to append to later
	mode_number = np.array([])
	growth_rate = np.array([])

	base_path = f"/home/userfs/l/lcv510/pedestal/results/{name}"

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

	plt.plot(mode_number, growth_rate, ".")
	plt.plot(mode_number, growth_rate, "--", alpha=0.4)
	plt.xlabel("Toroidal Mode Number, n") ; plt.ylabel("Growth Rate " r"($\gamma$)")
	plt.title(f"Growth Rate for a given Mode number for the {name} testcase")
	plt.show()

	print(f"Max growth rate: {np.max(growth_rate)}")



if __name__ == "__main__":

	action = input("Drive? y/n ")
	if action == "y":
		driver()
	elif action=="n":
		action2 = input("Extract? y/n ")
		if action2 == "y":
			extractor()
		

		



		



