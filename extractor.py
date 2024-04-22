import os
import numpy as np
import matplotlib.pyplot as plt

# 0) Name of testcase
name = input("Testace name: ")

# 1) Make empty array list to append to later
mode_number = np.array([])
growth_rate = np.array([])

# 2) Establish a root path to the current directory
basepath = "/home/userfs/l/lcv510/pedestal/Results/circb_results"

# 3) Enter the subdirectories
for folder, subfolder, files in os.walk(basepath):
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


# 8) Plot
plt.plot(mode_number, growth_rate, ".")
plt.plot(mode_number, growth_rate, "--", alpha=0.4)
plt.xlabel("Toroidal Mode Number, n") ; plt.ylabel("Growth Rate " r"($\gamma$)")
plt.title(f"Growth Rate for a given Mode number for the {name} testcase")
plt.show()
