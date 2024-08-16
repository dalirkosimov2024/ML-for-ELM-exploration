import os
import numpy as np
import shutil
import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
import subprocess
from pathlib2 import Path
import sys
import csv
from csv import writer 

def runner(delta, kappa, name):
    freeGS_runner(delta,kappa)
    ELITE_runner(name)
    max_gamma = extractor(name)
    gamma_appender(max_gamma)


def gamma_appender(max_gamma):
    path = "/home/userfs/l/lcv510/pedestal/results/ml-for-elms"
    with open(f"{path}/y_train.csv","a") as f:
        f.write("\n")
        writer_object = writer(f)
        writer_object.writerow([max_gamma])
        f.close()

def freeGS_runner(delta, kappa):

    freegs_path = "/home/userfs/l/lcv510/pedestal/freegs/metal_wall"
    os.chdir(freegs_path)
    delta = str(delta); kappa = str(kappa)
    print(f"\n \n Running delta {delta}")
    subprocess.run(["python3", "metal_wall.py", delta, kappa])
    print(f"\n \n Completed delta {delta}")

def ELITE_runner(name):
    name = name.replace('[', '')
    name = name.replace(']', '')
    path = "/home/userfs/l/lcv510/pedestal/ELITE/testcases/metal-wall/delta_sweep"
    new_path = f"{path}/{name}"
    print(f"\n name: {name}")
    os.chdir(new_path)
    print("\n Running ELITE")
    ELITE(name)


		
		
def replacer(filename, new_nn):
	file = Path(filename)

	filedata = file.read_text()

	nn_target = "nn="
	nmlow_target = "nmlow="
	nmvac_target = "nmvac="
	nmwinhalf_target = "nmwinhalf="

	res1 = filedata.split(nn_target, 1)
	res2 = filedata.split(nmlow_target, 1)
	res3 = filedata.split(nmvac_target,1)
	res4 = filedata.split(nmwinhalf_target, 1)

	nn = int("{}{}".format(res1[1][0], res1[1][1]))
	nmlow = int("{}{}".format(res2[1][0], res2[1][1]))
	nmvac = int("{}{}".format(res3[1][0], res3[1][1]))
	nmwinhalf = int("{}{}".format(res4[1][0], res4[1][1]))


	print(nn)
	print(f"{nmlow:02d}")
	print(nmvac)
	print(nmwinhalf)

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


def ELITE(name):

	#name = input("Testcase name: ")
	#eqin_condition = input("Select file type (eqdsk/eqin/dskbal): ")

	parent_path = "/scratch/lcv510/kappa_sweep/data"
	base_path = f"/{parent_path}/{name}"
	os.chdir(parent_path)
	if os.path.exists(base_path):
		os.chdir(base_path)
	else:
		os.mkdir(name)
		os.chdir(base_path)
	
	eq = f"~/pedestal/ELITE/ELITE16/equil/eliteeq -r {name}  "
	vac = f"~/pedestal/ELITE/ELITE16/vac/elitevac -r {name}"
	elite = f"~/pedestal/ELITE/ELITE16/symplas/elite -r {name} -I 1"
	
	nn_list = [5,10,15,20,25,30]

	print(os.getcwd(),"\n")


	for nn in nn_list:

		os.system("export OMP_NUM_THREADS=56")
		
		os.chdir(base_path)
		
		dir_name = f"nn_{nn}"
		os.mkdir(dir_name)
		new_dir = os.path.join(base_path, dir_name)
		
		shutil.copy("/home/userfs/l/lcv510/pedestal/ELITE/testcases/metal-wall/metal-wall.in",f"{new_dir}/{name}.in")
		
		shutil.copy(f"/scratch/lcv510/kappa_sweep/inputs/{name}/{name}.eqdsk",new_dir)
	
		os.chdir(new_dir)

		line_array  = np.array([])
		for filename in os.listdir(new_dir):
			if filename.endswith(".in"):
				replacer(filename, nn)

		print("\n\n\n ----- eq starting ----- \n\n\n")
		os.system(eq)
		print("\n\n\n ----- eq finished ----- \n\n\n")

		print("\n\n\n ----- vac starting ----- \n\n\n")
		os.system(vac)
		print("\n\n\n ----- vac finished ----- \n\n\n")


		print("\n\n\n ----- elite starting ----- \n\n\n")
		os.system(elite)
		print("\n\n\n ----- elite finished ----- \n\n\n")


def extractor(name):
    name = name.replace('[', '')
    name = name.replace(']', '')
	# 0) Name of testcase
	
	# 1) Make empty array list to append to later
    mode_number = np.array([])
    growth_rate = np.array([])
    
    base_path = f"/scratch/lcv510/kappa_sweep/data/{name}"

    # 3) Enter the subdirectories
    for folder, subfolder, files in os.walk(base_path):
            for file in files:
                    
                    # 4) Look for the fole containing the 
                    #    growth rates, in ".ghamma" files
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
                            print(f"Growth rate: {line_array[10]}")

                            # 7) Append desired values from the line array into
                            #    our initial empty array
                            mode_number = np.append(mode_number, float(line_array[9]))
                            growth_rate = np.append(growth_rate, float(line_array[10]))

    print(mode_number)
    """"
    plt.plot(sorted(mode_number), growth_rate[np.argsort(mode_number)], ".")
    plt.plot(sorted(mode_number), growth_rate[np.argsort(mode_number)], "--", alpha=0.4)
    plt.xlabel("Toroidal Mode Number, n") ; plt.ylabel("Growth Rate " r"($\gamma$)")
    plt.title(f"Growth Rate for a given Mode number for the {name} testcase")
    plt.show()
    max_growth_rate = np.max(growth_rate)
    print(f"Max growth rate: {max_growth_rate}")
    """
    max_growth_rate = np.max(growth_rate)

    return max_growth_rate

if __name__ == "__main__":

    kappa =1.2
    delta_list = [0, 0.2, 0.4, 0.6, 0.8] 
    for delta in delta_list:
        name = f"delta_{delta}"
        runner(delta, kappa, name)



		

