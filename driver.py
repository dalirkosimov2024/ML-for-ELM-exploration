import os
import shutil

copy_path = "/home/userfs/l/lcv510/pedestal/ELITE/testcases/circb/circb"
paste_path =  "/home/userfs/l/lcv510/pedestal/Results/circb_results"

eq = "~/pedestal/ELITE/ELITE16/equil/eliteeq"
vac = "~/pedestal/ELITE/ELITE16/vac/elitevac"
elite = "~/pedestal/ELITE/ELITE16/symplas/elite"

os.chdir(paste_path)

os.system(eq)
os.system(vac)
os.system(elite)
