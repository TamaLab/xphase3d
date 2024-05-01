# -*- coding: utf-8 -*-


# The main program for phase retrieval using xphase3dpy


import sys
import time
import datetime
import numpy as np
from xphase3dpy import optimize
from xphase3dpy import fileio


def print_usage():
    print ("")
    print ("Usage: python run.py sample.config")
    print ("")
    print ("Argument")
    print ("    - sample.config : Configuration file")
    print ("")


# By default, the initial model consists of random numbers
#  [0, 1) across the entire volume
def create_R0(xN, yN, zN):
    # Value range [0, 1)
    return np.random.rand(xN, yN, zN)


# By default, the support extends across the entire volume
def create_S(xN, yN, zN):
    return np.ones((xN, yN, zN), dtype=bool)


# By default. there is no missing part in Fourier space
def create_H(xN, yN, zN):
    return np.zeros((xN, yN, zN), dtype=bool)


# Obtain current time
def get_time():
    return datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")


def main(filename_config, mpi=False):
    # Start
    time_0 = time.time()
    # Creat a log file
    if mpi == True:
        filename_log = filename_config + ".log"
        try:
            sys.stdout = open(filename_log, "a")
        except Exception as e:
            print ("Error in creating {0}".format(filename_log))
            print ("    {0}".format(e))
            return 1
    print ("XPHASE3DPY - RUN")
    print ("")
    print ("{0:50} {1}".format("Program started", get_time()))
    print ("{0:50} {1}".format("Start reading parameters", get_time()))
    # Read and print parameters
    flag, config_dict = fileio.read_config(filename_config)
    if flag != 0:
        return 1
    print ("")
    print ("#############################################################"
           "#########")
    print ("CONFIG: {0}".format(filename_config))
    xN = int(config_dict["XN"])
    yN = int(config_dict["YN"])
    zN = int(config_dict["ZN"])
    need_R0 = int(config_dict["NEED R0"])
    need_S = int(config_dict["NEED S"])
    need_H = int(config_dict["NEED H"])
    keyword_M = config_dict["KEYWORD M"]
    keyword_Rp = config_dict["KEYWORD Rp"]
    keyword_R0 = ""
    if need_R0 == 1:
        keyword_R0 = config_dict["KEYWORD R0"]
    keyword_S = ""
    if need_S == 1:
        keyword_S = config_dict["KEYWORD S"]
    keyword_H = ""
    if need_H == 1:
        keyword_H = config_dict["KEYWORD H"]
    method = config_dict["METHOD"]
    beta =float(config_dict["BETA"])
    lower_bound = float(config_dict["LOWER BOUND"])
    upper_bound = float(config_dict["UPPER BOUND"])
    num_loop = int(config_dict["NUM LOOP"])
    num_iteration = int(config_dict["NUM ITERATION"])
    sigma0 = float(config_dict["SIGMA0"])
    sigmar = float(config_dict["SIGMAR"])
    th = float(config_dict["TH"])
    print ("")
    print ("Parameters:")
    print ("")
    print ("    xN:            {0}".format(xN))
    print ("    yN:            {0}".format(yN))
    print ("    zN:            {0}".format(zN))
    print ("")
    print ("    need_R0:       {0}".format(need_R0))
    print ("    need_S:        {0}".format(need_S))
    print ("    need_H:        {0}".format(need_H))
    print ("")
    print ("    keyword M:     {0}".format(keyword_M))
    print ("    keyword R0:    {0}".format(keyword_R0))
    print ("    keyword S:     {0}".format(keyword_S))
    print ("    keyword H:     {0}".format(keyword_H))
    print ("    keyword Rp:    {0}".format(keyword_Rp))
    print ("")
    print ("    method:        {0}".format(method))
    print ("    beta:          {0}".format(beta))
    print ("    lower_bound:   {0}".format(lower_bound))
    print ("    upper_bound:   {0}".format(upper_bound))
    print ("")
    print ("    num_loop:      {0}".format(num_loop))
    print ("    num_iteration: {0}".format(num_iteration))
    print ("")
    print ("    sigma0:        {0}".format(sigma0))
    print ("    sigmar:        {0}".format(sigmar))
    print ("    th:            {0}".format(th))
    print ("############################################################"
           "##########")
    print ("")
    # Read M
    print ("{0:50} {1}".format("Start reading M", get_time()))
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
    # Read or create R0
    print ("{0:50} {1}".format("Start reading/creating R0", get_time()))
    if need_R0 == 1:
        try:
            R0 = np.load(keyword_R0)
            if R0.shape != (xN, yN, zN):
                print ("Error in reading {0}".format(keyword_R0))
                print ("    require dimensions {0} * {1} * {2}"
                       .format(xN, yN, zN))
                return 1
        except Exception as e:
            print ("Error in reading {0}".format(keyword_R0))
            print ("    {0}".format(e))
            return 1
    else:
        R0 = create_R0(xN, yN, zN)
    # Read or create S
    print ("{0:50} {1}".format("Start reading/creating S", get_time()))
    if need_S == 1:
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
    else:
        S = create_S(xN, yN, zN)
    # Read of create H
    print ("{0:50} {1}".format("Start reading/creating H", get_time()))
    if need_H == 1:
        try:
            H = np.load(keyword_H)
            if H.shape != (xN, yN, zN):
                print ("Error in reading {0}".format(keyword_H))
                print ("    require dimensions {0} * {1} * {2}"
                       .format(xN, yN, zN))
                return 1
        except Exception as e:
            print ("Error in reading {0}".format(keyword_H))
            print ("    {0}".format(e))
            return 1
    else:
        H = create_H(xN, yN, zN)
    # Start loops
    print ("")
    print ("{0:50} {1}".format("Optimization starts", get_time()))
    R = np.copy(R0)
    time_1 = time.time()
    Rp = optimize.run_loop(R, M, S, H, num_loop, num_iteration, lower_bound,
                          upper_bound, method, beta, th, sigma0, sigmar)
    time_2 = time.time()
    # Save Rp
    print ("")
    print ("{0:50} {1}".format("Start writing Rp", get_time()))
    try:
        np.save(keyword_Rp, Rp)
    except Exception as e:
        print ("Error in writing {0}".format(keyword_Rp))
        print ("    {0}".format(e))
        return 1
    # Print execution time
    print ("{0:50} {1}".format("Task completed", get_time()))
    time_ini = time_1 - time_0
    time_opt = time_2 - time_1
    print ("")
    print ("############################################################"
           "##########")
    print ("Summary")
    print ("")
    print ("    xN:                          {0}".format(xN))
    print ("    yN:                          {0}".format(yN))
    print ("    zN:                          {0}".format(zN))
    print ("    method:                      {0}".format(method))
    print ("    num_loop:                    {0}".format(num_loop))
    print ("    num_iteration:               {0}".format(num_iteration))
    print ("    Time for initialization (s): {0:.6f}".format(time_ini))
    print ("    Time for optimization (s):   {0:.6f}".format(time_opt))
    print ("############################################################"
           "##########")
    print ("")
    if mpi == True:
        sys.stdout.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    else:
        filename_config = sys.argv[1]
        flag = main(filename_config)
        if flag != 0:
            sys.exit(1)


