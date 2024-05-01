#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fftw3-mpi.h"
#include "fileio.h"


// To assess the Fourier Shell Correlation (FSC) between two 3D models


void print_usage()
{
    printf ("\n");
    printf ("Usage: mpiexec -n <num_processes> fsc IN1 IN2 OUT XN YN ZN\n\n");
    printf ("Arguments\n");
    printf ("    - mpiexec -n <num_processes> : Number of MPI processes\n");
    printf ("    - IN1                        : Keyword of input file 1: 3D density model\n");
    printf ("    - IN2                        : Keyword of input file 2: 3D density model\n");
    printf ("    - OUT                        : Filename of the output FSC data\n");
    printf ("    - XN                         : Size of the X dimension\n");
    printf ("    - YN                         : Size of the Y dimension\n");
    printf ("    - ZN                         : Size of the Z dimension\n");
}


int main(int argc, char **argv)
{
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 7)
    {
        if (rank == 0)
        {
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    char* keyword_R1  = argv[1];
    char* keyword_R2  = argv[2];
    char* filename_FSC = argv[3];

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
        strcpy(filename_log, filename_FSC);
        strcat(filename_log, ".log");
        file_log = fopen(filename_log, "a");
        if (file_log == NULL)
        {
            printf ("Error in creating %s\n", filename_log);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        freopen(filename_log, "a", stdout);

        printf ("XPHASE3D - FSC\n");
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
        printf ("%-50s %s\n", "Start creating FFT plans", time_str);
        printf ("\n");
        fflush(stdout);
    }

    // Density models R1 and R2 and their Fourier transforms
    fftw_complex *local_R1   = fftw_alloc_complex(local_alloc);
    fftw_complex *local_F_R1 = fftw_alloc_complex(local_alloc);
    fftw_complex *local_R2   = fftw_alloc_complex(local_alloc);
    fftw_complex *local_F_R2 = fftw_alloc_complex(local_alloc);

    fftw_plan plan_fft_R1, plan_fft_R2;
    plan_fft_R1 = fftw_mpi_plan_dft_3d(xN, yN, zN, local_R1, local_F_R1, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_fft_R2 = fftw_mpi_plan_dft_3d(xN, yN, zN, local_R2, local_F_R2, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading IN1", time_str);
        fflush(stdout);
    }

    // Read R1
    int flag;
    double *local_R_real = (double*)malloc(sizeof(double) * local_indexN);
    flag = read_3d_double(keyword_R1, x_start, x_end, yN, zN, local_R_real);
    if (flag != 0)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_R1[i][0] = local_R_real[i];
        local_R1[i][1] = 0.0;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading IN2", time_str);
        printf ("\n");
        fflush(stdout);
    }

    // Read R2
    flag = read_3d_double(keyword_R2, x_start, x_end, yN, zN, local_R_real);
    if (flag != 0)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_R2[i][0] = local_R_real[i];
        local_R2[i][1] = 0.0;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start FSC calculation", time_str);
        printf ("\n");
        fflush(stdout);
    }

    // Fourier transform
    fftw_execute(plan_fft_R1);
    fftw_execute(plan_fft_R2);

    ptrdiff_t cenX, cenY, cenZ;
    // Integer division
    cenX = xN / 2;
    cenY = yN / 2;
    cenZ = zN / 2;
    ptrdiff_t cen_min = cenX;
    if (cenY < cen_min)    cen_min = cenY;
    if (cenZ < cen_min)    cen_min = cenZ;
    ptrdiff_t kN = cen_min + 1;

    // SUM { REAL ( F_R1 * conjugate(F_R2) ) }
    double *local_F1F2 = (double*)malloc(sizeof(double) * kN);
    // SUM { [ ABSOLUTE (F_R1) ] ^2 }
    double *local_F1SQ = (double*)malloc(sizeof(double) * kN);
    // SUM { [ ABSOLUTE (F_R2) ] ^2 }
    double *local_F2SQ = (double*)malloc(sizeof(double) * kN);
    for (ptrdiff_t i = 0; i < kN; i++)
    {
        local_F1F2[i] = 0.0;
        local_F1SQ[i] = 0.0;
        local_F2SQ[i] = 0.0;
    }

    // Sum (average) over k shells
    ptrdiff_t local_index, k;
    double k_double;
    // High-frequency components at located at the center after FFT
    // Use xf, yf, zf to shift low-frequency components to the center
    ptrdiff_t xf, yf, zf;
    for (ptrdiff_t x = x_start; x < x_end; x++)
    {
        xf = (x + cenX) % xN;
        for (ptrdiff_t y = 0; y < yN; y++)
        {
            yf = (y + cenY) % yN;
            for (ptrdiff_t z = 0; z < zN; z++)
            {
                zf = (z + cenZ) % zN;
                local_index = (x - x_start) * yN * zN + y * zN + z;
                k_double = sqrt((double)((xf-cenX)*(xf-cenX) + (yf-cenY)*(yf-cenY) + (zf-cenZ)*(zf-cenZ)));
                // 0.0 --> 0; (0, 1.0] --> 1; (1.0, 2.0] --> 2
                k = (ptrdiff_t)ceil(k_double);
                if (0 <= k && k < kN)
                {
                    local_F1F2[k] += local_F_R1[local_index][0] * local_F_R2[local_index][0]
                                   + local_F_R1[local_index][1] * local_F_R2[local_index][1];

                    local_F1SQ[k] += local_F_R1[local_index][0] * local_F_R1[local_index][0]
                                   + local_F_R1[local_index][1] * local_F_R1[local_index][1];

                    local_F2SQ[k] += local_F_R2[local_index][0] * local_F_R2[local_index][0]
                                   + local_F_R2[local_index][1] * local_F_R2[local_index][1];
                }
            }
        }
    }

    // Sum over segmented blocks to Process 0
    double *F1F2 = NULL;
    double *F1SQ = NULL;
    double *F2SQ = NULL;
    if (rank == 0)
    {
        F1F2 = (double*)malloc(sizeof(double) * kN);
        F1SQ = (double*)malloc(sizeof(double) * kN);
        F2SQ = (double*)malloc(sizeof(double) * kN);
    }
    MPI_Reduce(local_F1F2, F1F2, kN, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_F1SQ, F1SQ, kN, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_F2SQ, F2SQ, kN, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // FSC calculation
    if (rank == 0)
    {
        double *FSC = (double*)malloc(sizeof(double) * kN);
        for (ptrdiff_t i = 0; i < kN; i++)
        {
            if (F1SQ[i] > 2e-14 && F2SQ[i] > 2e-14)
            {
                FSC[i] = F1F2[i] / sqrt(F1SQ[i] * F2SQ[i]);
            }
            else
            {
                FSC[i] = 0.0;
            }
        }

        get_time_str(time_str);
        printf ("%-50s %s\n", "Start writing OUT", time_str);
        fflush(stdout);

        flag = write_profile(filename_FSC, kN, FSC);

        get_time_str(time_str);
        printf ("%-50s %s\n", "Task completed", time_str);
        printf ("\n");
        fflush(stdout);
        fclose(file_log);

        free(F1F2);
        free(F1SQ);
        free(F2SQ);
        free(FSC);
    }

    free(time_str);
    free(local_R1);
    free(local_R2);
    free(local_F_R1);
    free(local_F_R2);
    free(local_R_real);
    free(local_F1F2);
    free(local_F1SQ);
    free(local_F2SQ);
    fftw_destroy_plan(plan_fft_R1);
    fftw_destroy_plan(plan_fft_R2);

    fftw_mpi_cleanup();
    MPI_Finalize();

    return 0;
}
