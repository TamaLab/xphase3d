# -*- coding: utf-8 -*-


# The main mpi program for phase retrieval using xphase3dpy


import sys
from mpi4py import MPI
from xphase3dpy import run


def print_usage():
    print ("")
    print ("Usage: mpiexec -n <num_processes> python mpi_run.py sample_#.config")
    print ("")
    print ("Argument")
    print ("    - sample_#.config : Keyword of configuration files")
    print ("                        Use '#' as wildcard for rank number")


def main(keyword_config):
    # Check keyword_config
    count_hash = keyword_config.count('#')
    if count_hash != 1:
        if rank == 0:
            if count_hash == 0:
                print ("Error: {0} has no '#'".format(keyword_config))
            else:
                print ("Error: {0} has more than one '#'".
                       format(keyword_config))
        return 1
    path_config = keyword_config.replace('#', str(rank))
    # Run
    return run.main(path_config, mpi=True)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if len(sys.argv) != 2:
        if rank == 0:
            print_usage()
        sys.exit(1)
    else:
        keyword_config = sys.argv[1]
        flag = main(keyword_config)
        if flag != 0:
            sys.exit(1)
    MPI.Finalize()


