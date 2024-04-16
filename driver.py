import os
import shutil
import subprocess

in_copy_path = "/home/userfs/l/lcv510/pedestal/ELITE/testcases/circb/circb.in"
paste_path = "/home/userfs/l/lcv510/pedestal/Results/circb_results"
eq_copy_path = "/home/userfs/l/lcv510/pedestal/ELITE/testcases/circb/circb.eq"

eliteeq = "~/pedestal/ELITE/ELITE16/equil/eliteeq"
elitevac = "~/pedestal/ELITE/ELITE16/vac/elitevac"
elite = "~/pedestal/ELITE/ELITE16/equil/elite"

shutil.copy(in_copy_path, paste_path)
shutil.copy(eq_copy_path, paste_path)

os.chdir(paste_path)
os.system("~/pedestal/ELITE/ELITE16/equil/eliteeq")

