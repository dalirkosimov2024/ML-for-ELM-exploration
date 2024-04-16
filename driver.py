import os
import shutil

in_copy_path = "/home/userfs/l/lcv510/pedestal/ELITE/testcases/circb/circb.in"
in_paste_path = "/home/userfs/l/lcv510/pedestal/Results/circb_results"

eq_copy_path = "/home/userfs/l/lcv510/pedestal/ELITE/testcases/circb/circb.eq"
eq_paste_path = "/home/userfs/l/lcv510/pedestal/Results/circb_results"

shutil.copy(in_copy_path, in_paste_path)
shutil.copy(eq_copy_path, eq_paste_path)
