# -*- coding: utf-8 -*-


# A demo of using xphase3dpy


import subprocess
import numpy as np
from xphase3dpy import commands
from xphase3dpy import run
from xphase3dpy import tools

xN = 64
yN = 64
zN = 64

keyword_R = "0_data/R.npy"
keyword_M = "0_data/M.npy"
keyword_S = "0_data/S.npy"
keyword_R0 = "0_data/R0.npy"
keyword_H = "0_data/H.npy"
keyword_Rp_0 = "0_data/Rp_0.npy"
keyword_Rp_0_align = "0_data/Rp_0_align.npy"
keyword_Rp_1 = "0_data/Rp_1.npy"
keyword_Rp_1_align = "0_data/Rp_1_align.npy"
keyword_Rp_2 = "0_data/Rp_2.npy"
keyword_Rp_2_align = "0_data/Rp_2_align.npy"
keyword_Rp_012 = "0_data/Rp_012.npy"

print ("Check R\n")
R = np.load(keyword_R)
tools.save_mrc("0_data/R.mrc", R)

print ("Create M\n")
commands.make_m(keyword_R, keyword_M, xN, yN, zN)
M = np.load(keyword_M)
M_shift = np.roll(M, shift=(int(xN/2), int(yN/2), int(zN/2)), axis=(0,1,2))
tools.save_mrc("0_data/M_shift.mrc", M_shift)

print ("Create S\n")
sigma = 3.0
th = 0.05
commands.make_s(keyword_M, keyword_S, xN, yN, zN, sigma, th)
S = np.load(keyword_S)
tools.save_mrc("0_data/S.mrc", S)

print ("Create R0\n")
commands.make_r0(keyword_S, keyword_R0, xN, yN, zN)
R0 = np.load(keyword_R0)
tools.save_mrc("0_data/R0.mrc", R0)

print ("Create H\n")
radius = -1
H_shift = np.zeros((xN, yN, zN), dtype=bool)
cenX = int(xN/2)
cenY = int(yN/2)
cenZ = int(zN/2)
for dx in range(-10,10):
    for dy in range(-10,10):
        for dz in range(-10,10):
            r = (dx**2 + dy**2 + dz**2) ** 0.5
            if r <= radius:
                H_shift[cenX+dx, cenY+dy, cenZ+dz] = True
H = np.roll(H_shift, shift=(-cenX, -cenY, -cenZ), axis=(0,1,2))
np.save(keyword_H, H)
tools.save_mrc("0_data/H_shift.mrc", H_shift)

print ("MPI phase retrieval reconstructions: Rp_0 and Rp_1")
print ("(Log file: xx.config.log)\n")
# Enter "mpiexec -n 2 python xphase3dpy/mpi_run.py 0_demo_mpi_#.config"
#  in shell, or execute the following two lines
process = subprocess.Popen(
    "mpiexec -n 2 python -m xphase3dpy.mpi_run 0_demo_mpi_#.config",
    shell=True)
process.wait()

print ("Single-process phase retrieval reconstruction: Rp_2")
# Option 1: Enter "python xphase3dpy/run.py 0_demo_2.config" in shell,
#           or execute the following two lines
process = subprocess.Popen("python -m xphase3dpy.run 0_demo_2.config",
                           shell=True)
process.wait()
# Option 2: Call run.main()
# run.main("demo_single.config")

print ("Align Rp_0")
commands.align(keyword_R, keyword_Rp_0, keyword_Rp_0_align, xN, yN, zN)
Rp_0_align = np.load(keyword_Rp_0_align)
tools.save_mrc("0_data/Rp_0_align.mrc", Rp_0_align)

print ("\nAlign Rp_1")
commands.align(keyword_R, keyword_Rp_1, keyword_Rp_1_align, xN, yN, zN)
Rp_1_align = np.load(keyword_Rp_1_align)
tools.save_mrc("0_data/Rp_1_align.mrc", Rp_1_align)

print ("\nAlign Rp_2")
commands.align(keyword_R, keyword_Rp_2, keyword_Rp_2_align, xN, yN, zN)
Rp_2_align = np.load(keyword_Rp_2_align)
tools.save_mrc("0_data/Rp_2_align.mrc", Rp_2_align)

print ("\nMerge Rp_0, Rp_1, and Rp_2\n")
commands.merge("0_data/Rp_#_align.npy", 0, 2, keyword_Rp_012, xN, yN, zN)
Rp_merge = np.load(keyword_Rp_012)
tools.save_mrc("0_data/Rp_012.mrc", Rp_merge)

print ("Calculate PRTF between Rp_0, Rp_1, and Rp_2\n")
commands.prtf("0_data/Rp_#_align.npy", 0, 2, "0_data/prtf.dat", xN, yN, zN)

print ("Calculate FSC between R and Rp_012\n")
commands.fsc(keyword_R, keyword_Rp_012, "0_data/fsc.dat", xN, yN, zN)

# Preparation for 1_demo.sh
##################################################
print ("Segment R\n")
tools.segment_f(keyword_R, xN, yN, zN)
process = subprocess.Popen("mv 0_data/R_*.h5 1_data/", shell=True)
process.wait()

print ("Segment H\n")
tools.segment_b(keyword_H, xN, yN, zN)
process = subprocess.Popen("mv 0_data/H_*.h5 1_data/", shell=True)
process.wait()


