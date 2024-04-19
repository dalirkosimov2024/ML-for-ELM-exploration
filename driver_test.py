import os
import numpy as np

base_path = "/home/userfs/l/lcv510/pedestal/Results/circb_results"
os.chdir(base_path)
print(os.getcwd())

line_array = np.array([])
for filename in os.listdir(base_path):
	if filename.endswith(".in"):
		print(filename)

		g = open(filename, "r")
		for line in g:
			line = line.strip()
			columns = np.array(line.split())
			print(columns)
			line_array = np.append(line_array, columns)

		g.close()
		print(line_array[13])

