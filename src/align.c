#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fftw3-mpi.h"
#include "fileio.h"


// To align two 3D density models: Translational shift and possible conjugate flip
// The method is described in Appendix C, Zhao et al., IUCrJ, 2024


void print_usage()
{
    printf ("\n");
    printf ("Usage: mpiexec -n <num_processes> align IN1 IN2 OUT XN YN ZN\n\n");
    printf ("Arguments\n");
    printf ("    - mpiexec -n <num_processes> : Number of MPI processes\n");
    printf ("    - IN1                        : Keyword of input file 1: Density model of the reference\n");
    printf ("    - IN2                        : Keyword of input file 2: Density model to be aligned\n");
    printf ("    - OUT                        : Keyword of output file:  Density model IN2 after alignment\n");
    printf ("    - XN                         : Size of the X dimension\n");
    printf ("    - YN                         : Size of the Y dimension\n");
    printf ("    - ZN                         : Size of the Z dimension\n");
    printf ("\n");
}


int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 7)
    {
        if (rank == 0)
        {
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    char* keyword_R1 = argv[1];
    char* keyword_R2 = argv[2];
    char* keyword_R3 = argv[3];

    ptrdiff_t xN, yN, zN;
    char *endptr_xN, *endptr_yN, *endptr_zN;
    xN = strtol(argv[4], &endptr_xN, 10);
    yN = strtol(argv[5], &endptr_yN, 10);
    zN = strtol(argv[6], &endptr_zN, 10);
    if (*endptr_xN != '\0' || *endptr_yN != '\0' || *endptr_zN != '\0')
    {
        if (rank == 0)
        {
            printf ("\n");
            if (*endptr_xN != '\0')    printf ("Error: XN <%s> is not a valid integer\n", argv[4]);
            if (*endptr_yN != '\0')    printf ("Error: YN <%s> is not a valid integer\n", argv[5]);
            if (*endptr_zN != '\0')    printf ("Error: ZN <%s> is not a valid integer\n", argv[6]);
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
        strcpy(filename_log, keyword_R3);
        strcat(filename_log, ".log");
        file_log = fopen(filename_log, "a");
        if (file_log == NULL)
        {
            printf ("Error in creating %s\n", filename_log);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        freopen(filename_log, "a", stdout);

        printf ("XPHASE3D - ALIGN\n");
        printf ("\n");
        get_time_str(time_str);
        printf ("%-50s %s\n", "Program started", time_str);
        printf ("\n");
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
        printf ("%-50s %s\n", "Start creating FFT/IFFT plans", time_str);
        printf ("\n");
        fflush(stdout);
    }

    // R1: Density model of the reference (IN1) and its Fourier transform
    fftw_complex *local_R1       = fftw_alloc_complex(local_alloc);
    fftw_complex *local_F_R1     = fftw_alloc_complex(local_alloc);

    // R2: Density model to be aligned (IN2) and its Fourier transform
    fftw_complex *local_R2       = fftw_alloc_complex(local_alloc);
    fftw_complex *local_F_R2     = fftw_alloc_complex(local_alloc);

    // R2_f: Flip of the density model to be aligned (IN2) and its Fourier transform
    fftw_complex *local_R2_f     = fftw_alloc_complex(local_alloc);
    fftw_complex *local_F_R2_f   = fftw_alloc_complex(local_alloc);

    // Convolution of R1 & R2 and its Fourier transform
    fftw_complex *local_conv     = fftw_alloc_complex(local_alloc);
    fftw_complex *local_F_conv   = fftw_alloc_complex(local_alloc);

    // Convolution of R1 & R2_f and its Fourier transform
    fftw_complex *local_conv_f   = fftw_alloc_complex(local_alloc);
    fftw_complex *local_F_conv_f = fftw_alloc_complex(local_alloc);

    fftw_plan plan_fft_R1, plan_fft_R2, plan_fft_R2_f;
    fftw_plan plan_ifft_conv, plan_ifft_conv_f;

    plan_fft_R1      = fftw_mpi_plan_dft_3d(xN, yN, zN, local_R1, local_F_R1,
        MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE);

    plan_fft_R2      = fftw_mpi_plan_dft_3d(xN, yN, zN, local_R2, local_F_R2,
        MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE);

    plan_fft_R2_f    = fftw_mpi_plan_dft_3d(xN, yN, zN, local_R2_f, local_F_R2_f,
        MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE);

    plan_ifft_conv   = fftw_mpi_plan_dft_3d(xN, yN, zN, local_F_conv, local_conv,
        MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

    plan_ifft_conv_f = fftw_mpi_plan_dft_3d(xN, yN, zN, local_F_conv_f, local_conv_f,
        MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading IN1", time_str);
        fflush(stdout);
    }

    // Read R1
    int flag;
    double *local_R1_real = (double*)malloc(sizeof(double) * local_indexN);
    flag = read_3d_double(keyword_R1, x_start, x_end, yN, zN, local_R1_real);
    if (flag != 0)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_R1[i][0] = local_R1_real[i];
        local_R1[i][1] = 0.0;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading IN2 (1st time)", time_str);
        fflush(stdout);
    }

    // Read R2
    // To perform fft convolution, R2 needs to be right-shifted for (xN//2+1, yN//2+1, zN//2+1)
    ptrdiff_t x2, y2, z2, local_index2;
    ptrdiff_t x, y, z, local_index;
    double *slice = (double*)malloc(sizeof(double) * yN * zN);
    char *filename_R2 = (char*)malloc(sizeof(char) * 500);
    for (x2 = x_start; x2 < x_end; x2++)
    {
        x = (x2 - (xN/2+1) + xN) % xN;
        sprintf (filename_R2, "%s_%ld.h5", keyword_R2, x);
        flag = read_2d_double(filename_R2, yN, zN, slice);
        if (flag != 0)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (y2 = 0; y2 < yN; y2++)
        {
            y = (y2 - (yN/2+1) + yN) % yN;
            for (z2 = 0; z2 < zN; z2++)
            {
                z = (z2 - (zN/2+1) + zN) % zN;
                local_index2 = (x2 - x_start) * yN * zN + y2 * zN + z2;
                local_index = y * zN + z;
                local_R2[local_index2][0] = slice[local_index];
                local_R2[local_index2][1] = 0.0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading IN2 (2nd time)", time_str);
        printf ("\n");
        fflush(stdout);
    }

    // Read R2_f from R2
    // To perform fft convolution, R2_f needs to be right-shifted for (xN//2+1, yN//2+1, zN//2+1)
    for (x2 = x_start; x2 < x_end; x2++)
    {
        x = (xN - 1 - x2 + (xN/2+1)) % xN;
        sprintf (filename_R2, "%s_%ld.h5", keyword_R2, x);
        flag = read_2d_double(filename_R2, yN, zN, slice);
        if (flag != 0)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (y2 = 0; y2 < yN; y2++)
        {
            y = (yN - 1 - y2 + (yN/2+1)) % yN;
            for (z2 = 0; z2 < zN; z2++)
            {
                z = (zN - 1 - z2 + (zN/2+1)) % zN;
                local_index2 = (x2 - x_start) * yN * zN + y2 * zN + z2;
                local_index = y * zN + z;
                local_R2_f[local_index2][0] = slice[local_index];
                local_R2_f[local_index2][1] = 0.0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start alignment", time_str);
        fflush(stdout);
    }

    // Fourier transform
    fftw_execute(plan_fft_R1);
    fftw_execute(plan_fft_R2);
    fftw_execute(plan_fft_R2_f);

    // Multiplication in Fourier space
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_F_conv[i][0] = local_F_R1[i][0] * local_F_R2[i][0] - local_F_R1[i][1] * local_F_R2[i][1];
        local_F_conv[i][1] = local_F_R1[i][0] * local_F_R2[i][1] + local_F_R1[i][1] * local_F_R2[i][0];
        local_F_conv_f[i][0] = local_F_R1[i][0] * local_F_R2_f[i][0] - local_F_R1[i][1] * local_F_R2_f[i][1];
        local_F_conv_f[i][1] = local_F_R1[i][0] * local_F_R2_f[i][1] + local_F_R1[i][1] * local_F_R2_f[i][0];
    }

    // Inverse Fourier transform
    fftw_execute(plan_ifft_conv);
    fftw_execute(plan_ifft_conv_f);

    // Find the maximum in conv and conv_f
    double local_max_conv, local_max_conv_f;
    double max_conv, max_conv_f;
    ptrdiff_t local_max_x, local_max_y, local_max_z;
    ptrdiff_t local_max_x_f, local_max_y_f, local_max_z_f;

    // Local maximum
    local_max_conv = -1000.0;
    local_max_conv_f = -1000.0;
    for (x = x_start; x < x_end; x++)
    {
        for (y = 0; y < yN; y++)
        {
            for (z = 0; z < zN; z++)
            {
                local_index = (x - x_start) * yN * zN + y * zN + z;
                if (local_conv[local_index][0] > local_max_conv)
                {
                    local_max_conv = local_conv[local_index][0];
                    local_max_x = x;
                    local_max_y = y;
                    local_max_z = z;
                }
                if (local_conv_f[local_index][0] > local_max_conv_f)
                {
                    local_max_conv_f = local_conv_f[local_index][0];
                    local_max_x_f = x;
                    local_max_y_f = y;
                    local_max_z_f = z;
                }
            }
        }
    }

    // Global maximum
    MPI_Allreduce(&local_max_conv,    &max_conv,    1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_conv_f,  &max_conv_f,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Find and broadcast which process the global maximum comes from
    double *all_local_max_conv   = NULL;
    double *all_local_max_conv_f = NULL;
    if (rank == 0)
    {
        all_local_max_conv   = (double*)malloc(sizeof(double) * size);
        all_local_max_conv_f = (double*)malloc(sizeof(double) * size);
    }
    MPI_Gather(&local_max_conv,   1, MPI_DOUBLE, all_local_max_conv,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_max_conv_f, 1, MPI_DOUBLE, all_local_max_conv_f, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int rank_max, rank_max_f;
    if (rank == 0)
    {
        for (int rank_i = 0; rank_i < size; rank_i++)
        {
            if (all_local_max_conv[rank_i] == max_conv)
            {
                rank_max = rank_i;
            }
            if (all_local_max_conv_f[rank_i] == max_conv_f)
            {
                rank_max_f = rank_i;
            }
        }
        free(all_local_max_conv);
        free(all_local_max_conv_f);
    }
    MPI_Bcast(&rank_max,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rank_max_f, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Check if a conjugate flip is present
    // Note: Convolution inherently involves a flip operation
    int flag_f;
    // Find and broadcast the coordinates of the global maximum in two convolutions
    ptrdiff_t max_x, max_y, max_z;
    if (max_conv_f >= max_conv)
    {
        flag_f = 0;
        max_x = local_max_x_f;
        max_y = local_max_y_f;
        max_z = local_max_z_f;
        MPI_Bcast(&max_x, 1, MPI_AINT, rank_max_f, MPI_COMM_WORLD);
        MPI_Bcast(&max_y, 1, MPI_AINT, rank_max_f, MPI_COMM_WORLD);
        MPI_Bcast(&max_z, 1, MPI_AINT, rank_max_f, MPI_COMM_WORLD);
    }
    else
    {
        flag_f = 1;
        max_x = local_max_x;
        max_y = local_max_y;
        max_z = local_max_z;
        MPI_Bcast(&max_x, 1, MPI_AINT, rank_max, MPI_COMM_WORLD);
        MPI_Bcast(&max_y, 1, MPI_AINT, rank_max, MPI_COMM_WORLD);
        MPI_Bcast(&max_z, 1, MPI_AINT, rank_max, MPI_COMM_WORLD);
    }

    // Translational shift (after the optional flip)
    ptrdiff_t shift_x, shift_y, shift_z;
    shift_x = max_x - (xN / 2);
    shift_y = max_y - (yN / 2);
    shift_z = max_z - (zN / 2);

    if (rank == 0)
    {
        printf ("\n");
        printf ("######################################################################\n");
        printf ("FLIP: %-8d SHIFT_X: %-8ld SHIFT_Y: %-8ld SHIFT_Z: %-8ld\n", flag_f, shift_x, shift_y, shift_z);
        printf ("######################################################################\n");
        printf ("\n");
        fflush(stdout);
    }

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start writing OUT", time_str);
        fflush(stdout);
    }

    // Move R2 or R2_f
    char *filename_R3 = (char*)malloc(sizeof(char) * 500);
    ptrdiff_t x3, y3, z3, local_index3;
    for (x2 = x_start; x2 < x_end; x2++)
    {
        x = (x2 - (xN/2+1) + xN) % xN;
        x3 = (x + shift_x + xN) % xN;
        sprintf (filename_R3, "%s_%ld.h5", keyword_R3, x3);
        for (y2 = 0; y2 < yN; y2++)
        {
            y = (y2 - (yN/2+1) + yN) % yN;
            y3 = (y + shift_y + yN) % yN;
            for (z2 = 0; z2 < zN; z2++)
            {
                z = (z2 - (zN/2+1) + zN) % zN;
                z3 = (z + shift_z + zN) % zN;
                local_index2 = (x2 - x_start) * yN * zN + y2 * zN + z2;
                local_index3 = y3 * zN + z3;
                if (flag_f == 0)
                {
                    slice[local_index3] = local_R2[local_index2][0];
                }
                else
                {
                    slice[local_index3] = local_R2_f[local_index2][0];
                }
            }
        }
        flag = write_2d_double(filename_R3, yN, zN, slice);
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
    free(filename_R2);
    free(filename_R3);
    free(slice);
    free(local_R1_real);
    free(local_R1);
    free(local_R2);
    free(local_R2_f);
    free(local_conv);
    free(local_conv_f);
    free(local_F_R1);
    free(local_F_R2);
    free(local_F_R2_f);
    free(local_F_conv);
    free(local_F_conv_f);
    fftw_destroy_plan(plan_fft_R1);
    fftw_destroy_plan(plan_fft_R2);
    fftw_destroy_plan(plan_fft_R2_f);
    fftw_destroy_plan(plan_ifft_conv);
    fftw_destroy_plan(plan_ifft_conv_f);

    fftw_mpi_cleanup();
    MPI_Finalize();

    return 0;
}
