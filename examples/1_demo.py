# -*- coding: utf-8 -*-


# A continuation of 1_demo.sh


import subprocess
import numpy as np
from xphase3dpy import tools

xN = 64
yN = 64
zN = 64

print ("Check Rp_0, Rp_1, and Rp_2\n")

tools.connect("1_data/Rp_0_align.npy", xN, yN, zN)
R0 = np.load("1_data/Rp_0_align.npy")
tools.save_mrc("1_data/Rp_0_align.mrc", R0)

tools.connect("1_data/Rp_1_align.npy", xN, yN, zN)
R1 = np.load("1_data/Rp_1_align.npy")
tools.save_mrc("1_data/Rp_1_align.mrc", R1)

tools.connect("1_data/Rp_2_align.npy", xN, yN, zN)
R2 = np.load("1_data/Rp_2_align.npy")
tools.save_mrc("1_data/Rp_2_align.mrc", R2)


print ("Check Rp_0_align_bin4")
Rp_0_bin4 = tools.load_h5("1_data/Rp_0_align_bin4.h5")
tools.save_mrc("1_data/Rp_0_align_bin4.mrc", Rp_0_bin4)

