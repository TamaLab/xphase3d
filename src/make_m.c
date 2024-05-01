#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fftw3-mpi.h"
#include "fileio.h"


// To create the Fourier modulus (M) of a 3D density model (R)


void print_usage()
{
    printf ("\n");
    printf ("Usage: mpiexec -n <num_processes> make_m IN OUT XN YN ZN\n\n");
    printf ("Arguments\n");
    printf ("    - mpiexec -n <num_processes> : Number of MPI processes\n");
    printf ("    - IN                         : Keyword of input file: Density model\n");
    printf ("    - OUT                        : Keyword of output file: Fourier modulus\n");
    printf ("    - XN                         : Size of the X dimension\n");
    printf ("    - YN                         : Size of the Y dimension\n");
    printf ("    - ZN                         : Size of the Z dimension\n");
    printf ("\n");
}


int main(int argc, char **argv)
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

    char* keyword_R = argv[1];
    char* keyword_M = argv[2];

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
    if (rank == 0)
    {
        char filename_log[500];
        strcpy(filename_log, keyword_M);
        strcat(filename_log, ".log");
        file_log = fopen(filename_log, "a");
        if (file_log == NULL)
        {
            printf ("Error in creating %s\n", filename_log);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        freopen(filename_log, "a", stdout);

        printf ("XPHASE3D - MAKE_M\n");
        printf ("\n");
        get_time_str(time_str);
        printf ("%-50s %s\n", "Program started", time_str);
        fflush(stdout);
    }

    // Initialize fftw_mpi
    fftw_mpi_init();

    // Split 3D volume along X-axis
    // local_alloc can be larger than local_indexN
    ptrdiff_t local_alloc, local_xN, x_start, x_end, local_indexN;
    local_alloc = fftw_mpi_local_size_3d(xN, yN, zN, MPI_COMM_WORLD, &local_xN, &x_start);
    x_end = x_start + local_xN;
    local_indexN = local_xN * yN * zN;

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start creating FFT plan", time_str);
        fflush(stdout);
    }

    // Density model R and its Fourier transform F
    fftw_complex *local_R, *local_F;
    local_R = fftw_alloc_complex(local_alloc);
    local_F = fftw_alloc_complex(local_alloc);
    fftw_plan plan_fft;
    plan_fft = fftw_mpi_plan_dft_3d(xN, yN, zN, local_R, local_F, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading IN", time_str);
        fflush(stdout);
    }

    // Read R
    int flag;
    double *local_R_real = (double*)malloc(sizeof(double) * local_indexN);
    flag = read_3d_double(keyword_R, x_start, x_end, yN, zN, local_R_real);
    if (flag != 0)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_R[i][0] = local_R_real[i];
        local_R[i][1] = 0.0;
    }

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start calculation (FFT)", time_str);
        fflush(stdout);
    }

    // Fourier transform
    fftw_execute(plan_fft);

    // Fourier modulus M
    double *local_M = (double*)malloc(sizeof(double) * local_indexN);
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_M[i] = sqrt(local_F[i][0]*local_F[i][0] + local_F[i][1]*local_F[i][1]);
    }

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start writing OUT", time_str);
        fflush(stdout);
    }

    // Write M
    flag = write_3d_double(keyword_M, x_start, x_end, yN, zN, local_M);
    if (flag != 0)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Task completed", time_str);
        printf ("\n");
        fflush(stdout);
        fclose(file_log);
    }

    free(time_str);
    free(local_R);
    free(local_F);
    free(local_R_real);
    free(local_M);
    fftw_destroy_plan(plan_fft);

    fftw_mpi_cleanup();
    MPI_Finalize();

    return 0;
}
