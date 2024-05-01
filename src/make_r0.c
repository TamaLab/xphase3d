#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "fftw3-mpi.h"
#include "fileio.h"


// To create a random initial model (R0) based on the support (S)


void print_usage()
{
    printf ("\n");
    printf ("Usage: mpiexec -n <num_processes> make_r0 IN OUT XN YN ZN\n\n");
    printf ("Arguments\n");
    printf ("    - mpiexec -n <num_processes> : Number of MPI processes\n");
    printf ("    - IN                         : Keyword of input file: Support S\n");
    printf ("    - OUT                        : Keyword of output file: Initial model R0\n");
    printf ("    - XN                         : Size of the X dimension\n");
    printf ("    - YN                         : Size of the Y dimension\n");
    printf ("    - ZN                         : Size of the Z dimension\n");
    printf ("\n");
}


int main(int argc, char** argv)
{
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 6)
    {
        if (rank == 0)
        {
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    char* keyword_S = argv[1];
    char* keyword_R0 = argv[2];

    ptrdiff_t xN, yN, zN;
    char *endptr_xN, *endptr_yN, *endptr_zN;
    xN = strtol(argv[3], &endptr_xN, 10);
    yN = strtol(argv[4], &endptr_yN, 10);
    zN = strtol(argv[5], &endptr_zN, 10);
    if (*endptr_xN != '\0' || *endptr_yN != '\0' || *endptr_zN != '\0')
    {
        if (rank == 0)
        {
            printf ("\n");
            if (*endptr_xN != '\0')    printf ("Error: XN <%s> is not a valid integer\n", argv[3]);
            if (*endptr_yN != '\0')    printf ("Error: YN <%s> is not a valid integer\n", argv[4]);
            if (*endptr_zN != '\0')    printf ("Error: ZN <%s> is not a valid integer\n", argv[5]);
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    // Create a log file
    FILE *file_log;
    char *time_str = (char*)malloc(100 * sizeof(char));
    char filename_log[500];
    strcpy(filename_log, keyword_R0);
    strcat(filename_log, ".log");
    if (rank == 0)
    {
        file_log = fopen(filename_log, "a");
        if (file_log == NULL)
        {
            printf ("Error in creating %s\n", filename_log);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    freopen(filename_log, "a", stdout);
    if (rank == 0)
    {
        printf ("XPHASE3D - MAKE_R0\n");
        printf ("\n");
        get_time_str(time_str);
        printf ("%-50s %s\n", "Program started", time_str);
        printf ("\n");
        fflush(stdout);
    }

    // Initialize fftw_mpi
    fftw_mpi_init();

    // Split 3D volume along X-axis
    ptrdiff_t local_alloc, local_xN, x_start, x_end;
    local_alloc = fftw_mpi_local_size_3d(xN, yN, zN, MPI_COMM_WORLD, &local_xN, &x_start);
    local_alloc += 0;
    x_end = x_start + local_xN;

    // Create a random initial model within the support; value range [0, 1)
    srand(time(NULL));
    int flag = 0;
    char *filename_S  = (char*)malloc(sizeof(char) * 500);
    uint8_t *slice_S  = (uint8_t*)malloc(sizeof(uint8_t) * yN * zN);
    char *filename_R0 = (char*)malloc(sizeof(char) * 500);
    double *slice_R0  = (double*)malloc(sizeof(double) * yN * zN);
    for (ptrdiff_t x = x_start; x < x_end; x++)
    {
        sprintf (filename_S, "%s_%ld.h5", keyword_S, x);
        flag = read_2d_uint8(filename_S, yN, zN, slice_S);
        if (flag != 0)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (ptrdiff_t i = 0; i < yN * zN; i++)
        {
            if (slice_S[i] != 0)
            {
                slice_R0[i] = (double)rand() / RAND_MAX;
            }
            else
            {
                slice_R0[i] = 0.0;
            }
        }

        sprintf (filename_R0, "%s_%ld.h5", keyword_R0, x);
        flag = write_2d_double(filename_R0, yN, zN, slice_R0);
        if (flag != 0)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        get_time_str(time_str);
        printf ("%s %-39ld %s\n", "Slice NO. ", x, time_str);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf ("\n");
        get_time_str(time_str);
        printf ("%-50s %s\n", "Task completed", time_str);
        printf ("\n");
        fflush(stdout);
        fclose(file_log);
    }

    free(time_str);
    free(slice_S);
    free(slice_R0);
    free(filename_S);
    free(filename_R0);

    fftw_mpi_cleanup();
    MPI_Finalize();

    return 0;
}
