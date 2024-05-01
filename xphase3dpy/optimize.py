# -*- coding: utf-8 -*-


# Functions for projections and iterative optimizations in xphase3dpy


import datetime
import numpy as np
from scipy.fftpack import fftn, ifftn
from scipy.ndimage import gaussian_filter


# Projection to satisfy M constraint
# Return to complex Rpc
# Rpc is used in shrink wrapping process
def projM_c(R, M, H):
    F = fftn(R)
    P = np.angle(F)
    Fp = np.copy(M)
    Fp[H==True] = np.absolute(F[H==True])
    Fp = Fp * np.cos(P) + 1j * Fp * np.sin(P)
    # Rpc is complex
    Rpc = ifftn(Fp)
    return Rpc


# Projection to satisfy M constraint
def projM(R, M, H):
    return np.real(projM_c(R, M, H))


# Reflection by M constraint
def reflM(R, M, H):
    return 2 * projM(R, M, H) - R


# Projection to satisfy S constraint
def projS(R, S, lower_bound, upper_bound):
    Rp = np.copy(R)
    # Inside S, higher than the upper bound
    condition1 = (R > upper_bound) * (S==True)
    Rp[condition1] = upper_bound
    # Inside S, lower than the lower bound
    condition2 = (R < lower_bound) * (S==True)
    Rp[condition2] = lower_bound
    # Outside S
    condition3 = (S==False)
    Rp[condition3] = 0.0
    return Rp


# Reflection by S constraint
def reflS(R, S, lower_bound, upper_bound):
    return 2 * projS(R, S, lower_bound, upper_bound) - R


# Hybrid Input-Output (HIO)
def optHIO(R, M, S, H, lower_bound, upper_bound, beta):
    RpM = projM(R, M, H)
    R2 = np.copy(RpM)
    # Inside S, higher than the upper bound
    condition1 = (RpM > upper_bound) * (S==True)
    R2[condition1] = R[condition1] - beta*(RpM[condition1] - upper_bound)
    # Inside S, lower than the lower bound
    condition2 = (RpM < lower_bound) * (S==True)
    R2[condition2] = R[condition2] - beta*(RpM[condition2] - lower_bound)
    # Outside S
    condition3 = (S==False)
    R2[condition3] = R[condition3] - beta*RpM[condition3]
    return R2


# Error Reduction (ER)
# beta is not needed; to be consistent with other optimization methods
def optER(R, M, S, H, lower_bound, upper_bound, beta):
    RpM = projM(R, M, H)
    R2 = projS(RpM, S, lower_bound, upper_bound)
    return R2


# Averaged Successive Reflections (ASR)
# beta is not needed; to be consistent with other optimization methods
def optASR(R, M, S, H, lower_bound, upper_bound, beta):
    RrM = reflM(R, M, H)
    R2 = (reflS(RrM, S, lower_bound, upper_bound) + R) * 0.5
    return R2


# Hybrid Projection Reflection (HPR)
def optHPR(R, M, S, H, lower_bound, upper_bound, beta):
    RpM = projM(R, M, H)
    # RrM = reflM(R, M, H); to avoid unnecessary FFT
    RrM = 2 * RpM - R
    # Combination of rM and pM with beta
    Rb = RrM + (beta - 1.0) * RpM
    # rS of Rb
    RbrS = reflS(Rb, S, lower_bound, upper_bound)
    R2 = 0.5 * (RbrS + R + (1.0 - beta) * RpM)
    return R2


# Relaxed Averaged Alternating Reflectors (RAAR)
def optRAAR(R, M, S, H, lower_bound, upper_bound, beta):
    RpM = projM(R, M, H)
    # RrM = reflM(R, M, H); to avoid unnecessary FFT
    RrM = 2 * RpM - R
    # rS of RrM
    RrMrS = reflS(RrM, S, lower_bound, upper_bound)
    R2 = 0.5 * beta * (RrMrS + R) + (1 - beta) * RpM
    return R2


# Difference Map (DM)
def optDM(R, M, S, H, lower_bound, upper_bound, beta):
    # Other two parameters
    s = 1.0 / beta
    m = - 1.0 / beta
    # projM first, then linear combination with s, and finally projS
    RpM = projM(R, M, H)
    Rs = (1 + s) * RpM - s * R
    RspS = projS(Rs, S, lower_bound, upper_bound)
    # projS first, then linear combination with m, and finally projM
    RpS = projS(R, S, lower_bound, upper_bound)
    Rm = (1 + m) * RpS - m * R
    RmpM = projM(Rm, M, H)
    # R2 after optimization
    R2 = R + beta * RspS - beta * RmpM
    return R2


# Shrink wrap: update the support
def update_S(Rpc, th, sigma):
    cR = gaussian_filter(np.absolute(Rpc), sigma=sigma, order=0)
    threshold = np.max(cR) * th
    S = np.zeros(cR.shape, dtype=bool)
    S[cR > threshold] = True
    return S


# Obtain current time
def get_time():
    return datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")


# Run iterations without shrink wrap
def run_iteration(R, M, S, H, num_iteration, lower_bound, upper_bound,
                  method, beta):

    optFUN = {"ASR": optASR, "DM": optDM, "ER": optER,
              "HIO": optHIO, "HPR": optHPR, "RAAR": optRAAR}
    for iteration in range(num_iteration):
        print ("{0} {1:<36} {2}".format(
            "    iteration", iteration, get_time()))
        R = optFUN[method](R, M, S, H, lower_bound, upper_bound, beta)
    return R


# Run shrink wrap loops
def run_loop(R, M, S, H, num_loop, num_iteration, lower_bound, upper_bound,
             method, beta, th, sigma0, sigmar):

    for loop in range(num_loop):
        print ("")
        print ("{0} {1:<45} {2}".format("LOOP", loop, get_time()))
        R = run_iteration(R, M, S, H, num_iteration, lower_bound, upper_bound,
                      method, beta)
        Rpc = projM_c(R, M, H)
        sigma = sigma0 * (1 - sigmar)**loop
        S = update_S(Rpc, th, sigma)
        R = np.real(Rpc)
    return R


