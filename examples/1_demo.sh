#!/bin/bash


# A demo of using xphase3d


set -e

echo
echo "Create M"
mpiexec -n 2 bin/make_m 1_data/R 1_data/M 64 64 64

echo
echo "Create S"
mpiexec -n 2 bin/make_s 1_data/M 1_data/S 64 64 64 3.0 0.05

echo
echo "Create R0"
mpiexec -n 2 bin/make_r0 1_data/S 1_data/R0 64 64 64

echo
echo "Phase retrieval reconstructions: Rp_0, Rp_1, Rp_2"
mpiexec -n 2 bin/run 1_demo_0.config
mpiexec -n 2 bin/run 1_demo_1.config
mpiexec -n 2 bin/run 1_demo_2.config

echo
echo "Align Rp_0"
mpiexec -n 2 bin/align 1_data/R 1_data/Rp_0 1_data/Rp_0_align 64 64 64

echo
echo "Align Rp_1"
mpiexec -n 2 bin/align 1_data/R 1_data/Rp_1 1_data/Rp_1_align 64 64 64

echo
echo "Align Rp_2"
mpiexec -n 2 bin/align 1_data/R 1_data/Rp_2 1_data/Rp_2_align 64 64 64

echo
echo "Merge Rp_0, Rp_1, and Rp_2"
mpiexec -n 2 bin/merge 1_data/Rp_#_align 0 2 1_data/Rp_012 64 64 64

echo
echo "Calculate PRTF between Rp_0, Rp_1, and Rp_2"
mpiexec -n 2 bin/prtf 1_data/Rp_#_align 0 2 1_data/prtf.dat 64 64 64

echo
echo "Calculate FSC between R and Rp_012"
mpiexec -n 2 bin/fsc 1_data/R 1_data/Rp_012 1_data/fsc.dat 64 64 64

echo
echo "Bin S (datatype uint8) from 64^3 to 32^3"
./bin/bin_b 2 1_data/S 1_data/S_bin2.h5 64 64 64

echo
echo "Bin Rp_0 (datatype double) from 64^3 to 16^3"
./bin/bin_f 4 1_data/Rp_0_align 1_data/Rp_0_align_bin4.h5 64 64 64

echo
echo "Excute 1_demo.py to continue"
echo
python 1_demo.py
