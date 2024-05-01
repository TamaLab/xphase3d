# -*- coding: utf-8 -*-


# List of commands in xphase3dpy


import numpy as np
from scipy.fftpack import fftn, ifftn
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter


# To create the Fourier modulus (M) of a 3D density model (R)
def make_m(keyword_R, keyword_M, xN, yN, zN):
    # Read R
    try:
        R = np.load(keyword_R)
        if R.shape != (xN, yN, zN):
            print ("Error in reading {0}".format(keyword_R))
            print ("    require dimensions {0} * {1} * {2}"
                   .format(xN, yN, zN))
            return 1
    except Exception as e:
        print ("Error in reading {0}".format(keyword_R))
        print ("    {0}".format(e))
        return 1
    # Fourier transform
    M = np.absolute(fftn(R))
    # Write M
    try:
        np.save(keyword_M, M)
    except Exception as e:
        print ("Error in writing {0}".format(keyword_M))
        print ("    {0}".format(e))
        return 1
    return 0


# To create an initial support (S) based on the Fourier modulus (M)
# sigma: Standard deviation of Gaussian convolution
# th: Threshold factor
def make_s(keyword_M, keyword_S, xN, yN, zN, sigma, th):
    # Check sigma and th
    if sigma <= 0 or th <= 0:
        if sigma <= 0:
            print ("Error: sigma should be a positive number")
        if th <= 0:
            print ("Error: th should be a positive number")
        return 1
    # Read M
    try:
        M = np.load(keyword_M)
        if M.shape != (xN, yN, zN):
            print ("Error in reading {0}".format(keyword_M))
            print ("    require dimensions {0} * {1} * {2}"
                   .format(xN, yN, zN))
            return 1
    except Exception as e:
        print ("Error in reading {0}".format(keyword_M))
        print ("    {0}".format(e))
        return 1
    # Autocorrelation function (ACF) and its convolution (ACF_conv)
    ACF = np.real(ifftn(M**2))
    ACF_conv = gaussian_filter(ACF, sigma=sigma, order=0)
    # Support S
    threshold = np.max(ACF_conv) * th
    S = np.zeros(M.shape, dtype=bool)
    S[ACF_conv>threshold] = True
    # Right shift S for (cenX, cenY, cenZ)
    cenX = int(xN/2)
    cenY = int(yN/2)
    cenZ = int(zN/2)
    S = np.roll(S, shift=(cenX, cenY, cenZ), axis=(0,1,2))
    # Write S
    try:
        np.save(keyword_S, S)
    except Exception as e:
        print ("Error in writing {0}".format(keyword_S))
        print ("    {0}".format(e))
        return 1
    return 0


# To create a random initial model (R0) based on the support (S)
def make_r0(keyword_S, keyword_R0, xN, yN, zN):
    # Read S
    try:
        S = np.load(keyword_S)
        if S.shape != (xN, yN, zN):
            print ("Error in reading {0}".format(keyword_S))
            print ("    require dimensions {0} * {1} * {2}"
                   .format(xN, yN, zN))
            return 1
    except Exception as e:
        print ("Error in reading {0}".format(keyword_S))
        print ("    {0}".format(e))
        return 1
    # R0: Value range [0, 1)
    R0 = np.random.rand(int(xN), int(yN), int(zN))
    R0[S==False] = 0
    # Write R0
    try:
        np.save(keyword_R0, R0)
    except Exception as e:
        print ("Error in writing {0}".format(keyword_R0))
        print ("    {0}".format(e))
        return 1
    return 0


# To align two 3D density models: Translational shift and
#  possible conjugate flip
# The method is described in Appendix C, Zhao et al., IUCrJ, 2024
def align(keyword_R1, keyword_R2, keyword_R3, xN, yN, zN):
    # Read R1
    try:
        R1 = np.load(keyword_R1)
        if R1.shape != (xN, yN, zN):
            print ("Error in reading {0}".format(keyword_R1))
            print ("    require dimensions {0} * {1} * {2}"
                   .format(xN, yN, zN))
            return 1
    except Exception as e:
        print ("Error in reading {0}".format(keyword_R1))
        print ("    {0}".format(e))
        return 1
    # Read R2
    try:
        R2 = np.load(keyword_R2)
        if R2.shape != (xN, yN, zN):
            print ("Error in reading {0}".format(keyword_R2))
            print ("    require dimensions {0} * {1} * {2}"
                   .format(xN, yN, zN))
            return 1
    except Exception as e:
        print ("Error in reading {0}".format(keyword_R2))
        print ("    {0}".format(e))
        return 1
    # Convolution of R1 & R2
    CONV = convolve(R1, R2, mode="same", method="auto")
    CONV_max = np.max(CONV)
    # Convolution of R1 & R2_f
    R2_f = np.flip(R2, axis=(0,1,2))
    CONV_f = convolve(R1, R2_f, mode="same", method="auto")
    CONV_max_f = np.max(CONV_f)
    # Check if a conjugate flip is present
    # Note: Convolution inherently involves a flip operation
    if CONV_max >= CONV_max_f:
        flag_f = True
        max_pos = np.where(CONV==CONV_max)
    else:
        flag_f = False
        max_pos = np.where(CONV_f == CONV_max_f)
    # Translational shift (after the optional flip)
    max_x = max_pos[0][0]
    max_y = max_pos[1][0]
    max_z = max_pos[2][0]
    shift_x = max_x - int(xN/2)
    shift_y = max_y - int(yN/2)
    shift_z = max_z - int(zN/2)
    print ("FLIP: {0:<8} SHIFT_X: {1:<8} SHIFT_Y: {2:<8} SHIFT_Z: {3:<8}"
           .format(flag_f, shift_x, shift_y, shift_z))
    # Move R2 to R3
    if flag_f == False:
        R3 = np.copy(R2)
    else:
        R3 = np.copy(R2_f)
    R3 = np.roll(R3, shift=(shift_x, shift_y, shift_z), axis=(0,1,2))
    # Save R3
    try:
        np.save(keyword_R3, R3)
    except Exception as e:
        print ("Error in writing {0}".format(keyword_R3))
        print ("    {0}".format(e))
        return 1
    return 0


# To merge a series of aligned 3D density models
# keyword_in: Use '#' as wildcard for sequential number
def merge(keyword_in, start, end, keyword_out, xN, yN, zN):
    # Check keyword_in
    count_hash = keyword_in.count('#')
    if count_hash != 1:
        if count_hash == 0:
            print ("Error: {0} has no '#'".format(keyword_in))
        else:
            print ("Error: {0} has more than one '#'".format(keyword_in))
        return 1
    # Check start & end
    start = int(start)
    end = int(end)
    if start > end:
        print ("Error: end must not be smaller than start")
        return 1
    # Merge
    R_out = np.zeros((int(xN), int(yN), int(zN)), dtype=np.float64)
    for s in range(start, end+1):
        filename_in = keyword_in.replace('#', str(s))
        try:
            R_in = np.load(filename_in)
            if R_in.shape != (xN, yN, zN):
                print ("Error in reading {0}".format(filename_in))
                print ("    require dimensions {0} * {1} * {2}"
                       .format(xN, yN, zN))
                return 1
        except Exception as e:
            print ("Error in reading {0}".format(filename_in))
            print ("    {0}".format(e))
            return 1
        R_out = R_out + R_in
    R_out = R_out / (end + 1 - start)
    # Save R_out
    try:
        np.save(keyword_out, R_out)
    except Exception as e:
        print ("Error in writing {0}".format(keyword_out))
        print ("    {0}".format(e))
        return 1
    return 0


# To assess the Phase Retrieval Transfer Function (PRTF)
#  of a set of aligned 3D models
# keyword_R: Use '#' as wildcard for sequential number
def prtf(keyword_R, start, end, keyword_prtf, xN, yN, zN):
    # Check keyword_R
    count_hash = keyword_R.count('#')
    if count_hash != 1:
        if count_hash == 0:
            print ("Error: {0} has no '#'".format(keyword_R))
        else:
            print ("Error: {0} has more than one '#'".format(keyword_R))
        return 1
    # Check start & end
    start = int(start)
    end = int(end)
    if start > end:
        print ("Error: end must not be smaller than start")
        return 1
    # Average of Fourier transforms over models
    # Put low-frequency components at the center after Fourier transform
    xN = int(xN)
    yN = int(yN)
    zN = int(zN)
    cenX = int(xN/2)
    cenY = int(yN/2)
    cenZ = int(zN/2)
    F_cmplx = np.zeros((xN,yN,zN), dtype=np.complex128)
    F_abslt = np.zeros((xN,yN,zN), dtype=np.float64)
    for s in range(start, end+1):
        filename_R = keyword_R.replace('#', str(s))
        try:
            R = np.load(filename_R)
            if R.shape != (xN, yN, zN):
                print ("Error in reading {0}".format(filename_R))
                print ("    require dimensions {0} * {1} * {2}"
                       .format(xN, yN, zN))
                return 1
        except Exception as e:
            print ("Error in reading {0}".format(filename_R))
            print ("    {0}".format(e))
            return 1
        F = np.roll(fftn(R), shift=(cenX, cenY, cenZ), axis=(0,1,2))
        F_cmplx = F_cmplx + F
        F_abslt = F_abslt + np.absolute(F)
    F_cmplx = F_cmplx / (end + 1 - start)
    F_abslt = F_abslt / (end + 1 - start)
    # Average over k shells
    kN = min(cenX, cenY, cenZ) + 1
    PRTF = np.zeros(kN, dtype=np.float64)
    center = np.reshape([cenX, cenY, cenZ], (3,1,1,1))
    indices = np.indices((xN, yN, zN), dtype=np.float64) - center
    radius = (indices[0]**2 + indices[1]**2 + indices[2]**2) ** 0.5
    for k in range(0, kN):
        points = (k-1 < radius) * (radius <= k)
        denominator = np.average(F_abslt[points])
        if denominator > 2e-14:
            PRTF[k] = np.average(np.absolute(F_cmplx[points])) / denominator
        else:
            PRTF[k] = 0.0
    # Write PRTF
    try:
        with open(keyword_prtf, "w") as f:
            for k in range(0, kN):
                f.write("{0:<10}\t{1:.6f}\n".format(k, PRTF[k]))
    except Exception as e:
        print ("Error in reading {0}".format(keyword_prtf))
        print ("    {0}".format(e))
        return 1
    return 0


# To assess the Fourier Shell Correlation (FSC) between two 3D models
def fsc(keyword_R1, keyword_R2, keyword_fsc, xN, yN, zN):
    # Read R1
    try:
        R1 = np.load(keyword_R1)
        if R1.shape != (xN, yN, zN):
            print ("Error in reading {0}".format(keyword_R1))
            print ("    require dimensions {0} * {1} * {2}"
                   .format(xN, yN, zN))
            return 1
    except Exception as e:
        print ("Error in reading {0}".format(keyword_R1))
        print ("    {0}".format(e))
        return 1
    # Read R2
    try:
        R2 = np.load(keyword_R2)
        if R2.shape != (xN, yN, zN):
            print ("Error in reading {0}".format(keyword_R2))
            print ("    require dimensions {0} * {1} * {2}"
                   .format(xN, yN, zN))
            return 1
    except Exception as e:
        print ("Error in reading {0}".format(keyword_R2))
        print ("    {0}".format(e))
        return 1
    # Fourier transform of R1 and R2
    # Put low-frequency components at the center
    xN = int(xN)
    yN = int(yN)
    zN = int(zN)
    cenX = int(xN/2)
    cenY = int(yN/2)
    cenZ = int(zN/2)
    F1 = np.roll(fftn(R1), shift=(cenX, cenY, cenZ), axis=(0,1,2))
    F2 = np.roll(fftn(R2), shift=(cenX, cenY, cenZ), axis=(0,1,2))
    # Average over k shells
    kN = min(cenX, cenY, cenZ) + 1
    F1F2 = np.zeros(kN, dtype=np.float64)
    F1SQ = np.zeros(kN, dtype=np.float64)
    F2SQ = np.zeros(kN, dtype=np.float64)
    FSC = np.zeros(kN, dtype=np.float64)
    center = np.reshape([cenX, cenY, cenZ], (3,1,1,1))
    indices = np.indices((xN, yN, zN), dtype=np.float64) - center
    radius = (indices[0]**2 + indices[1]**2 + indices[2]**2) ** 0.5
    for k in range(0, kN):
        points = (k-1 < radius) * (radius <= k)
        F1F2[k] = np.average(np.real((F1[points] * np.conjugate(F2[points]))))
        F1SQ[k] = np.average(np.absolute(F1[points])**2)
        F2SQ[k] = np.average(np.absolute(F2[points])**2)
        if F1SQ[k] > 2e-14 and F2SQ[k] > 2e-14:
            FSC[k] = F1F2[k] / (F1SQ[k] * F2SQ[k])**0.5
        else:
            FSC[k] = 0.0
    # Write FSC
    try:
        with open(keyword_fsc, "w") as f:
            for k in range(0, kN):
                f.write("{0:<10}\t{1:.6f}\n".format(k, FSC[k]))
    except Exception as e:
        print ("Error in reading {0}".format(keyword_fsc))
        print ("    {0}".format(e))
        return 1
    return 0


