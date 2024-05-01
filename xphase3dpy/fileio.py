# -*- coding: utf-8 -*-


# Functions for reading and writing files in xphase3dpy


# Read parameters from a configuration file
def read_config(path_config):
    flag = 0
    ##################################################
    # Read
    config_dict = {}
    try:
        with open(path_config, "r") as f:
            lines = f.readlines()
            for line in lines:
                # Remove leading and trailing spaces and tabs
                line = line.strip()
                # Skip an empty line or a line that starts with '#'
                if len(line) != 0 and line[0] != "#" and line.count(":") == 1:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    config_dict[key] = value
    except Exception as e:
        print ("Error in reading {0}".format(path_config))
        print ("    {0}".format(e))
        flag = 1
        return flag, config_dict
    ##################################################
    # Check
    # List of required parameters
    key_list = ["XN", "YN", "ZN", "NEED R0", "NEED S", "NEED H", "KEYWORD M",
                "KEYWORD R0", "KEYWORD S", "KEYWORD H", "KEYWORD Rp",
                "METHOD", "BETA", "LOWER BOUND", "UPPER BOUND",
                "NUM LOOP", "NUM ITERATION", "SIGMA0", "SIGMAR", "TH"]
    # By default, set R0, S, and H as "needed"
    need_R0 = 1
    need_S = 1
    need_H = 1
    # List of available methods
    method_list = ["HIO", "ER", "ASR", "HPR", "RAAR", "DM"]
    # Check if the provided parameters are valid
    for key in config_dict.keys():
        # Check int
        if key in ["XN", "YN", "ZN", "NUM LOOP", "NUM ITERATION"]:
            try:
                int(config_dict[key])
            except:
                print ("    - {0} is not an integer".format(key))
                flag = 1
        # Check float
        elif key in ["BETA", "LOWER BOUND", "UPPER BOUND", "SIGMA0",
                     "SIGMAR", "TH"]:
            try:
                float(config_dict[key])
            except:
                print ("    - {0} is not a float number".format(key))
                flag = 1
        # Check 0 or 1
        elif key in ["NEED R0", "NEED S", "NEED H"]:
            try:
                if int(config_dict[key]) in [0, 1]:
                    if key == "NEED R0":
                        need_R0 = int(config_dict[key])
                    elif key == "NEED S":
                        need_S = int(config_dict[key])
                    elif key == "NEED H":
                        need_H = int(config_dict[key])
                else:
                    print ("    - {0} should be either 0 or 1".format(key))
                    flag = 1
            except:
                print ("    - {0} should be either 0 or 1".format(key))
                flag = 1
        # Check methods
        elif key == "METHOD":
            if config_dict[key] not in method_list:
                print ("    - METHOD {0} is not available"
                       .format(config_dict[key]))
                print ("    - Available methods:", end='')
                for method in method_list:
                    print (" {0}".format(method), end='')
                print ("")
                flag = 1
    # Check if any parameter is omitted
    for key in key_list:
        if key not in config_dict.keys():
            if key == "KEYWORD R0":
                if need_R0 == 1:
                    print ("    - {0} is not defined".format(key))
                    flag = 1
            elif key == "KEYWORD S":
                if need_S == 1:
                    print ("    - {0} is not defined".format(key))
                    flag = 1
            elif key == "KEYWORD H":
                if need_H == 1:
                    print ("    - {0} is not defined".format(key))
                    flag = 1
            else:
                print ("    - {0} is not defined".format(key))
                flag = 1
    ##################################################
    if flag == 1:
        print ("Error in reading parameters from {0}".format(path_config))
    return flag, config_dict


