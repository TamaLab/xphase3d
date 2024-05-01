#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fftw3-mpi.h"
#include "fileio.h"


// To assess the Phase Retrieval Transfer Function (PRTF) of a set of aligned 3D models


void print_usage()
{
    printf ("\n");
    printf ("Usage: mpiexec -n <num_processes> prtf IN START END OUT XN YN ZN\n\n");
    printf ("Arguments\n");
    printf ("    - mpiexec -n <num_processes> : Number of MPI processes\n");
    printf ("    - IN                         : Keyword of input 3D density models\n");
    printf ("                                   Use '#' as wildcard for sequential number\n");
    printf ("    - START                      : Starting sequential number\n");
    printf ("    - END                        : Ending sequential number\n");
    printf ("    - OUT                        : Filename of the output PRTF data\n");
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

    if (argc != 8)
    {
        if (rank == 0)
        {
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    char *keyword_in = argv[1];
    char *filename_PRTF = argv[4];
    int count_hash = 0;
    for (size_t i = 0; i < strlen(keyword_in); i++)
    {
        if (keyword_in[i] == '#')
        {
            count_hash++;
        }
    }
    if (count_hash != 1)
    {
        if (rank == 0)
        {
            printf ("\n");
            if (count_hash == 0)    printf ("Error: IN <%s> has no '#'\n", keyword_in);
            else         printf ("Error: IN <%s> has more than one '#'\n", keyword_in);
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    ptrdiff_t start, end, xN, yN, zN;
    char *endptr_start, *endptr_end, *endptr_xN, *endptr_yN, *endptr_zN;
    start = strtol(argv[2], &endptr_start, 10);
    end   = strtol(argv[3], &endptr_end,   10);
    xN    = strtol(argv[5], &endptr_xN,    10);
    yN    = strtol(argv[6], &endptr_yN,    10);
    zN    = strtol(argv[7], &endptr_zN,    10);

    if (*endptr_start != '\0' || *endptr_end != '\0'
        || *endptr_xN != '\0' || *endptr_yN != '\0' || *endptr_zN != '\0')
    {
        if (rank == 0)
        {
            printf ("\n");
            if (*endptr_start != '\0')    printf ("Error: START <%s> is not a valid integer\n", argv[2]);
            if (*endptr_end   != '\0')    printf ("Error: END <%s> is not a valid integer\n", argv[3]);
            if (*endptr_xN    != '\0')    printf ("Error: XN <%s> is not a valid integer\n", argv[4]);
            if (*endptr_yN    != '\0')    printf ("Error: YN <%s> is not a valid integer\n", argv[5]);
            if (*endptr_zN    != '\0')    printf ("Error: ZN <%s> is not a valid integer\n", argv[6]);
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    if (start > end)
    {
        if (rank == 0)
        {
            printf ("\n");
            printf ("Error: END must not be smaller than START\n");
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
        strcpy(filename_log, filename_PRTF);
        strcat(filename_log, ".log");
        file_log = fopen(filename_log, "a");
        if (file_log == NULL)
        {
            printf ("Error in creating %s\n", filename_log);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        freopen(filename_log, "a", stdout);

        printf ("XPHASE3D - PRTF\n");
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
        printf ("%-50s %s\n", "Start creating FFT plan", time_str);
        printf ("\n");
        fflush(stdout);
    }

    // Density model R and its Fourier transform
    fftw_complex *local_R   = fftw_alloc_complex(local_alloc);
    fftw_complex *local_F_R = fftw_alloc_complex(local_alloc);

    fftw_plan plan_fft_R;
    plan_fft_R = fftw_mpi_plan_dft_3d(xN, yN, zN, local_R, local_F_R, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

    // Sum (average) over models
    double *local_F_abs   = (double*)malloc(sizeof(double) * local_indexN);
    double *local_FC_abs  = (double*)malloc(sizeof(double) * local_indexN);
    double *local_FC_real = (double*)malloc(sizeof(double) * local_indexN);
    double *local_FC_imag = (double*)malloc(sizeof(double) * local_indexN);
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_F_abs[i]   = 0.0;
        local_FC_real[i] = 0.0;
        local_FC_imag[i] = 0.0;
    }

    int flag;
    double *local_R_real = (double*)malloc(sizeof(double) * local_indexN);
    char *keyword_in_s = (char*)malloc(sizeof(char) * 500);
    for (ptrdiff_t s = start; s < end + 1; s++)
    {
        if (rank == 0)
        {
            get_time_str(time_str);
            printf ("%s %-29ld %s\n", "Start reading IN NO.", s, time_str);
            fflush(stdout);
        }

        replace(keyword_in, keyword_in_s, s);
        flag = read_3d_double(keyword_in_s, x_start, x_end, yN, zN, local_R_real);
        if (flag != 0)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            get_time_str(time_str);
            printf ("%-50s %s\n", "    - start processing (FFT)", time_str);
            fflush(stdout);
        }

        for (ptrdiff_t i = 0; i < local_indexN; i++)
        {
            local_R[i][0] = local_R_real[i];
            local_R[i][1] = 0.0;
        }
        fftw_execute(plan_fft_R);

        for (ptrdiff_t i = 0; i < local_indexN; i++)
        {
            local_FC_real[i] += local_F_R[i][0];
            local_FC_imag[i] += local_F_R[i][1];
            local_F_abs[i]   += sqrt(local_F_R[i][0] * local_F_R[i][0] + local_F_R[i][1] * local_F_R[i][1]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        printf ("\n");
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start PRTF calculation", time_str);
        printf ("\n");
        fflush(stdout);
    }

    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_FC_abs[i] = sqrt(local_FC_real[i] * local_FC_real[i] + local_FC_imag[i] * local_FC_imag[i]);
    }

    // Sum (average) over k shells
    ptrdiff_t cenX, cenY, cenZ;
    // Integer division
    cenX = xN / 2;
    cenY = yN / 2;
    cenZ = zN / 2;
    ptrdiff_t cen_min = cenX;
    if (cenY < cen_min)    cen_min = cenY;
    if (cenZ < cen_min)    cen_min = cenZ;
    ptrdiff_t kN = cen_min + 1;
    double *local_FC_abs_k = (double*)malloc(sizeof(double) * kN);
    double *local_F_abs_k  = (double*)malloc(sizeof(double) * kN);
    for (ptrdiff_t i = 0; i < kN; i++)
    {
        local_FC_abs_k[i] = 0.0;
        local_F_abs_k[i]  = 0.0;
    }

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
            {
                for (ptrdiff_t z = 0; z < zN; z++)
                {
                    zf = (z + cenZ) % zN;
                    local_index = (x - x_start) * yN * zN + y * zN + z;
                    k_double = sqrt((double)((xf-cenX)*(xf-cenX) + (yf-cenY)*(yf-cenY) + (zf-cenZ)*(zf-cenZ)));
                    // 0.0 --> 0; (0, 1.0] --> 1; (1.0, 2.0] --> 2
                    k = (ptrdiff_t)ceil(k_double);
                    if (0 <= k && k < kN)
                    {
                        local_FC_abs_k[k] += local_FC_abs[local_index];
                        local_F_abs_k[k]  += local_F_abs[local_index];
                    }
                }
            }
        }
    }

    // Sum over segmented blocks to Process 0
    double *FC_abs_k = NULL;
    double *F_abs_k  = NULL;
    if (rank == 0)
    {
        FC_abs_k = (double*)malloc(sizeof(double) * kN);
        F_abs_k =  (double*)malloc(sizeof(double) * kN);
    }
    MPI_Reduce(local_FC_abs_k, FC_abs_k, kN, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_F_abs_k,  F_abs_k,  kN, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // PRTF calculation
    if (rank == 0)
    {
        double *PRTF = (double*)malloc(sizeof(double) * kN);
        for (ptrdiff_t i = 0; i < kN; i++)
        {
            if (fabs(local_F_abs_k[i]) > 2e-14)
            {
                PRTF[i] = FC_abs_k[i] / F_abs_k[i];
            }
            else
            {
                PRTF[i] = 0.0;
            }
        }

        get_time_str(time_str);
        printf ("%-50s %s\n", "Start writing OUT", time_str);
        fflush(stdout);

        flag = write_profile(filename_PRTF, kN, PRTF);

        get_time_str(time_str);
        printf ("%-50s %s\n", "Task completed", time_str);
        printf ("\n");
        fflush(stdout);
        fclose(file_log);

        free(FC_abs_k);
        free(F_abs_k);
        free(PRTF);
    }

    free(time_str);
    free(local_F_abs);
    free(local_FC_abs);
    free(local_FC_real);
    free(local_FC_imag);
    free(local_R_real);
    free(keyword_in_s);
    free(local_R);
    free(local_F_R);
    free(local_FC_abs_k);
    free(local_F_abs_k);
    fftw_destroy_plan(plan_fft_R);

    fftw_mpi_cleanup();
    MPI_Finalize();

    return 0;
}
