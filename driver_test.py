import os
import numpy as np
import shutil

name = input("Testcase name: ")

nn_list = np.array([])
for x in range(6,41):
	nn_list = np.append(nn_list,int(x))
print(nn_list,"\n")

eq = f"~/pedestal/ELITE/ELITE16/equil/eliteeq -r {name} "
vac = f"~/pedestal/ELITE/ELITE16/vac/elitevac -r {name}"
elite = f"~/pedestal/ELITE/ELITE16/symplas/elite -r {name} -I 0"


base_path = f"/home/userfs/l/lcv510/pedestal/Results/{name}"
os.chdir(base_path)
print(os.getcwd(),"\n")


for nn in nn_list:
	
	os.chdir(base_path)
	
	dir_name = f"nn_{nn}"
	os.mkdir(dir_name)
	new_dir = os.path.join(base_path, dir_name)
	
	shutil.copy(f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/{name}/{name}.in",new_dir)
	shutil.copy(f"/home/userfs/l/lcv510/pedestal/ELITE/testcases/{name}/{name}.dskbal",new_dir)
	os.chdir(new_dir)

	line_array  = np.array([])
	for filename in os.listdir(new_dir):
		if filename.endswith(".in"):
			g = open(filename, "r")
			filedata = g.read()
			g.close()

			newdata = filedata.replace("n= 6", f"n= {nn}")

			g = open(filename, "w")
			g.write(newdata)
			g.close()
	os.system(eq)
	os.system(vac)
	os.system(elite)
	



			#for line in g:
				#line = line.strip()
				#columns = np.array(line.split())
				#line_array = np.append(line_array, columns)
			#line_array[13] = nn
		



