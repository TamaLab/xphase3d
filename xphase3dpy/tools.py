# -*- coding: utf-8 -*-


# Additional tools for xphase3dpy and xphase3d


import numpy as np
import mrcfile
import h5py


# Save a numpy real 3D array as .mrc format
def save_mrc(filename, R):
    try:
        with mrcfile.new(filename, overwrite=True) as mrc:
            mrc.set_data(R.astype(np.float32))
    except Exception as e:
        print ("Error in writing {0}".format(filename))
        print ("    {0}".format(e))


# Read a real 3D array in .mrc format
def load_mrc(filename):
    try:
        with mrcfile.open(filename) as mrc:
            R = np.copy(mrc.data).astype(np.float64)
        return R
    except Exception as e:
            print ("Error in reading {0}".format(filename))
            print ("    {0}".format(e))


# Read numpy nd array in .h5 format ("/dataset")
def load_h5(filename):
    try:
        f = h5py.File(filename, "r")
        data = np.array(f["/dataset"])
        f.close()
        return data
    except Exception as e:
        print ("Error in reading {0}".format(filename))
        print ("    {0}".format(e))


# Segment a numpy 3d array (float) along the x-axis into h5 slices
#  required for mpi-xphase3d
def segment_f(filename, xN, yN, zN):
    try:
        R = np.load(filename)
        if R.shape != (xN, yN, zN):
            print ("Error in reading {0}".format(filename))
            print ("    require dimensions {0} * {1} * {2}"
                   .format(xN, yN, zN))
            return
    except Exception as e:
        print ("Error in reading {0}".format(filename))
        print ("    {0}".format(e))
        return
    R = R.astype(np.float64)
    # Little-endian
    dtype = R.dtype.newbyteorder('<')
    for x in range(0, int(xN)):
        filename_x = filename.replace(".npy", "_{0}.h5".format(x))
        R_x = R[x]
        try:
            f = h5py.File(filename_x, "w")
            f.create_dataset("dataset", shape=(int(yN), int(zN)),
                                       dtype=dtype, data=R_x)
            f.close()
        except Exception as e:
            print ("Error in writing {0}".format(filename))
            print ("    {0}".format(e))
            return


# Segment a numpy 3d array (bool) along the x-axis into h5 slices
#  required for mpi-xphase3d
def segment_b(filename, xN, yN, zN):
    try:
        H = np.load(filename)
        if H.shape != (xN, yN, zN):
            print ("Error in reading {0}".format(filename))
            print ("    require dimensions {0} * {1} * {2}"
                   .format(xN, yN, zN))
            return
    except Exception as e:
        print ("Error in reading {0}".format(filename))
        print ("    {0}".format(e))
        return
    H = H.astype(np.ubyte)
    for x in range(0, int(xN)):
        filename_x = filename.replace(".npy", "_{0}.h5".format(x))
        H_x = H[x]
        try:
            f = h5py.File(filename_x, "w")
            f.create_dataset("dataset", shape=(int(yN), int(zN)),
                                       dtype=np.ubyte, data=H_x)
            f.close()
        except Exception as e:
            print ("Error in writing {0}".format(filename))
            print ("    {0}".format(e))
            return


# Connect h5 slices along the x-axis to a numpy 3d array
def connect(filename, xN, yN, zN):
    # Determine the datatype based on the first slice
    x = 0
    filename_x = filename.replace(".npy", "_{0}.h5".format(x))
    try:
        f = h5py.File(filename_x, "r")
        R_x = np.array(f["/dataset"])
        if R_x.shape != (yN, zN):
            print ("Error in reading {0}".format(filename_x))
            print ("    require dimensions {0} * {1}".format(yN, zN))
            return
    except Exception as e:
        print ("Error in reading {0}".format(filename_x))
        print ("    {0}".format(e))
        return
    dtype = R_x.dtype
    # Connection
    R = np.zeros((int(xN), int(yN), int(zN)), dtype=dtype)
    for x in range(0, int(xN)):
        filename_x = filename.replace(".npy", "_{0}.h5".format(x))
        try:
            f = h5py.File(filename_x, "r")
            R_x = np.array(f["/dataset"])
            if R_x.shape != (yN, zN):
                print ("Error in reading {0}".format(filename_x))
                print ("    require dimensions {0} * {1}".format(yN, zN))
                return
        except Exception as e:
            print ("Error in reading {0}".format(filename_x))
            print ("    {0}".format(e))
            return
        R[x] = np.copy(R_x)
    # Write R
    try:
        np.save(filename, R)
    except Exception as e:
        print ("Error in writing {0}".format(filename))
        print ("    {0}".format(e))
        return


