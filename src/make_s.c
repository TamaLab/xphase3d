#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "fftw3-mpi.h"
#include "fileio.h"
#include "optimize.h"


// To create an initial support (S) based on the Fourier modulus (M)


void print_usage()
{
    printf ("\n");
    printf ("Usage: mpiexec -n <num_processes> make_s IN OUT XN YN ZN SIGMA TH\n\n");
    printf ("Arguments\n");
    printf ("    - mpiexec -n <num_processes> : Number of MPI processes\n");
    printf ("    - IN                         : Keyword of input file: Fourier modulus M\n");
    printf ("    - OUT                        : Keyword of output file: Support S\n");
    printf ("    - XN                         : Size of the X dimension\n");
    printf ("    - YN                         : Size of the Y dimension\n");
    printf ("    - ZN                         : Size of the Z dimension\n");
    printf ("    - SIGMA                      : Standard deviation of Gaussian convolution\n");
    printf ("    - TH                         : Threshold factor\n");
    printf ("\n");
}


int main(int argc, char **argv)
{
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 8)
    {
        if (rank == 0)
        {
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    char* keyword_M = argv[1];
    char* keyword_S = argv[2];

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

    double sigma, th;
    char *endptr_sg, *endptr_th;
    sigma = strtod(argv[6], &endptr_sg);
    th    = strtod(argv[7], &endptr_th);
    if (*endptr_sg != '\0' || *endptr_th != '\0')
    {
        if (rank == 0)
        {
            printf ("\n");
            if (*endptr_sg != '\0')    printf ("Error: SIGMA <%s> is not a float number\n", argv[6]);
            if (*endptr_th != '\0')    printf ("Error: TH <%s> is not a float number\n", argv[7]);
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    if (sigma <= 0 || th <= 0)
    {
        if (rank == 0)
        {
            printf ("\n");
            if (sigma <= 0)    printf ("Error: SIGMA <%f> is not a positive number\n", sigma);
            if (th <= 0)       printf ("Error: TH <%f> is not a positive number\n", th);
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
        strcpy(filename_log, keyword_S);
        strcat(filename_log, ".log");
        file_log = fopen(filename_log, "a");
        if (file_log == NULL)
        {
            printf ("Error in creating %s\n", filename_log);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        freopen(filename_log, "a", stdout);

        printf ("XPHASE3D - MAKE_S\n");
        printf ("\n");
        get_time_str(time_str);
        printf ("%-50s %s\n", "Program started", time_str);
        fflush(stdout);
    }

    // Initialize fftw_mpi
    fftw_mpi_init();

    // Split 3D volume along X-axis
    // local_alloc can be larger than local_indexN
    ptrdiff_t indexN, local_alloc, local_xN, x_start, x_end, local_indexN;
    indexN = xN * yN * zN;
    local_alloc = fftw_mpi_local_size_3d(xN, yN, zN, MPI_COMM_WORLD, &local_xN, &x_start);
    x_end = x_start + local_xN;
    local_indexN = local_xN * yN * zN;
    // t_wrap_for: runtime for FOR loops in shrink wrap
    double t_wrap_for;

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start creating FFT/IFFT plans", time_str);
        fflush(stdout);
    }

    // Autocorrelation function and its Fourier transform
    fftw_complex *local_acf, *local_F_acf;
    local_acf = fftw_alloc_complex(local_alloc);
    local_F_acf = fftw_alloc_complex(local_alloc);
    fftw_plan plan_ifft_acf, plan_fft_acf;
    plan_ifft_acf = fftw_mpi_plan_dft_3d(xN, yN, zN, local_F_acf, local_acf, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan_fft_acf  = fftw_mpi_plan_dft_3d(xN, yN, zN, local_acf, local_F_acf, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading IN", time_str);
        fflush(stdout);
    }

    // Read M
    int flag;
    double *local_M = (double*)malloc(sizeof(double) * local_indexN);
    flag = read_3d_double(keyword_M, x_start, x_end, yN, zN, local_M);
    if (flag != 0)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start calculation", time_str);
        fflush(stdout);
    }

    // Obtain ACF from the square of Fourier modulus and castaway imaginary parts
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_F_acf[i][0] = local_M[i] * local_M[i];
        local_F_acf[i][1] = 0.0;
    }
    fftw_execute(plan_ifft_acf);
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        // Normalization after inverse Fourier transform (optional)
        local_acf[i][0] = local_acf[i][0] / indexN;
        // Cast away imaginary parts (mandatory)
        local_acf[i][1] = 0.0;
    }
    fftw_execute(plan_fft_acf);

    // Gaussian kernel and its Fourier transform
    fftw_complex *local_kernel, *local_F_kernel;
    local_kernel = fftw_alloc_complex(local_alloc);
    local_F_kernel = fftw_alloc_complex(local_alloc);
    fftw_plan plan_fft_kernel;
    plan_fft_kernel = fftw_mpi_plan_dft_3d(xN, yN, zN, local_kernel, local_F_kernel, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    get_kernel(x_start, x_end, xN, yN, zN, sigma, local_kernel, &t_wrap_for);
    fftw_execute(plan_fft_kernel);

    // Convolution of the autocorrelation function and its Fourier transform
    fftw_complex *local_acf_conv, *local_F_acf_conv;
    local_acf_conv = fftw_alloc_complex(local_alloc);
    local_F_acf_conv = fftw_alloc_complex(local_alloc);
    fftw_plan plan_ifft_acf_conv;
    plan_ifft_acf_conv = fftw_mpi_plan_dft_3d(xN, yN, zN, local_F_acf_conv, local_acf_conv, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_F_acf_conv[i][0] = local_F_acf[i][0] * local_F_kernel[i][0] - local_F_acf[i][1] * local_F_kernel[i][1];
        local_F_acf_conv[i][1] = local_F_acf[i][0] * local_F_kernel[i][1] + local_F_acf[i][1] * local_F_kernel[i][0];
    }

    fftw_execute(plan_ifft_acf_conv);

    // Normalization after inverse Fourier transform (optional)
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_acf_conv[i][0] = local_acf_conv[i][0] / indexN;
    }

    // Support S
    uint8_t* local_S = (uint8_t*)malloc(sizeof(uint8_t) * local_indexN);
    update_S(th, local_indexN, local_S, local_acf_conv, MPI_COMM_WORLD, &t_wrap_for);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start writing OUT", time_str);
        fflush(stdout);
    }

    // Right shift S for (cenX, cenY, cenZ) to S2
    ptrdiff_t cenX, cenY, cenZ;
    cenX = xN / 2;
    cenY = yN / 2;
    cenZ = zN / 2;
    uint8_t* slice_S2 = (uint8_t*)malloc(sizeof(uint8_t) * yN * zN);
    char *filename_S2 = (char*)malloc(sizeof(char) * 500);
    ptrdiff_t x, y, z, local_index;
    ptrdiff_t x2, y2, z2, slice_index2;
    for (x = x_start; x < x_end; x++)
    {
        x2 = (x + cenX) % xN;
        sprintf (filename_S2, "%s_%ld.h5", keyword_S, x2);

        for (y = 0; y < yN; y++)
        {
            y2 = (y + cenY) % yN;
            for (z = 0; z < zN; z++)
            {
                z2 = (z + cenZ) % zN;
                local_index  = (x - x_start) * yN * zN + y * zN + z;
                slice_index2 = y2 * zN + z2;
                slice_S2[slice_index2] = local_S[local_index];
            }
        }

        flag = write_2d_uint8(filename_S2, yN, zN, slice_S2);
        if (flag != 0)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
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
    free(local_M);
    free(local_acf);
    free(local_kernel);
    free(local_F_acf);
    free(local_F_kernel);
    free(local_F_acf_conv);
    free(local_acf_conv);
    free(local_S);
    free(slice_S2);
    free(filename_S2);
    fftw_destroy_plan(plan_fft_acf);
    fftw_destroy_plan(plan_ifft_acf);
    fftw_destroy_plan(plan_fft_kernel);
    fftw_destroy_plan(plan_ifft_acf_conv);

    fftw_mpi_cleanup();
    MPI_Finalize();

    return 0;
}
