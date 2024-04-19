import os

base_path = "/home/userfs/l/lcv510/pedestal/Results/circb_results"
os.chdir(base_path)
print(os.getcwd())

for files in os.listdir(base_path):
	if files.endswith(".in"):
		print(files)

