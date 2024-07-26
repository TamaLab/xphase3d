#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "fftw3-mpi.h"
#include "fileio.h"
#include "optimize.h"


/*
The main program for phase retrieval using xphase3d
Version 2024.07

Contributors:
    - Wenyang Zhao
    - Osamu Miyashita
    - Miki Nakano
    - Florence Tama

Contact:
    - osamu.miyashita@riken.jp
    - florence.tama@riken.jp

At:
    Computational Structural Biology Research Team
    RIKEN Center for Computational Science
*/


void print_usage()
{
    printf ("\n");
    printf ("Usage: mpiexec -n <num_processes> run sample.config\n");
    printf ("\n");
    printf ("Arguments\n");
    printf ("    - mpiexec -n <num_processes> : Number of MPI processes\n");
    printf ("    - sample.config              : Configuration file\n");
    printf ("\n");
}


int create_R0(ptrdiff_t local_indexN, double *local_R0);
int create_S(ptrdiff_t local_indexN, uint8_t *local_S);
int create_H(ptrdiff_t local_indexN, uint8_t *local_H);
int initialize_RC(ptrdiff_t local_indexN, fftw_complex *local_RC, double *local_R0);
int run_iteration(fftw_plan plan_fft, fftw_plan plan_ifft,
    char *method, int num_iteration, double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H,
    double *t_iter_for, double *t_iter_fft);
int run_loop(int num_loop, double sigma0, double sigmar, double th,
    char *method, int num_iteration, double beta, double lower_bound, double upper_bound,
    ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t xN, ptrdiff_t yN, ptrdiff_t zN,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_plan plan_fft_kernel, fftw_plan plan_fft_R_abs, fftw_plan plan_ifft_R_conv,
    fftw_plan plan_fft, fftw_plan plan_ifft,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H,
    fftw_complex *local_kernel, fftw_complex *local_R_abs, fftw_complex *local_R_conv,
    fftw_complex *local_F_kernel, fftw_complex *local_F_R_abs, fftw_complex *local_F_R_conv,
    double *t_iter_for, double *t_iter_fft, double *t_wrap_for, double *t_wrap_fft);
//


// Global variable for MPI rank and size of mpi
int RANK, SIZE;

int main(int argc, char **argv)
{

    int flag;

    // Initialzie MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &SIZE);

    if (argc != 2)
    {
        if (RANK == 0)
        {
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    // To measure the running time
    char *time_str = (char*)malloc(100 * sizeof(char));
    clock_t c_0, c_1, c_2;
    double t_ini, t_opt;

    // t_iter_for: runtime for FOR loops in iterations
    // t_iter_fft: runtime for FFT in iterations
    // t_wrap_for: runtime for FOR loops in shrink wrap
    // t_wrap_fft: runtime for FFT in shrink wrap
    double t_iter_for, t_iter_fft, t_wrap_for, t_wrap_fft;
    t_iter_for = 0.0;
    t_iter_fft = 0.0;
    t_wrap_for = 0.0;
    t_wrap_fft = 0.0;

    c_0 = clock();

    // Create a log file
    FILE *file_log;
    if (RANK == 0)
    {
        char filename_log[500];
        strcpy(filename_log, argv[1]);
        strcat(filename_log, ".log");
        file_log = fopen(filename_log, "a");
        if (file_log == NULL)
        {
            printf ("Error in creating %s\n", filename_log);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        freopen(filename_log, "a", stdout);

        printf ("XPHASE3D - RUN\n");
        printf ("\n");
        get_time_str(time_str);
        printf ("%-50s %s\n", "Program started", time_str);
        fflush(stdout);
    }

    // Parameters to be read
    ptrdiff_t xN, yN, zN;
    int need_R0, need_S, need_H;
    char *keyword_M  = (char*)malloc(sizeof(char) * 500);
    char *keyword_R0 = (char*)malloc(sizeof(char) * 500);
    char *keyword_S  = (char*)malloc(sizeof(char) * 500);
    char *keyword_H  = (char*)malloc(sizeof(char) * 500);
    char *keyword_Rp = (char*)malloc(sizeof(char) * 500);
    char *method = (char*)malloc(sizeof(char) * 500);
    double beta, lower_bound, upper_bound;
    int num_loop, num_iteration;
    double sigma0, sigmar, th;

    if (RANK == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading parameters", time_str);
        fflush(stdout);
    }

    // Read parameters in Process 0
    // To avoid too many processes reading the same file
    if (RANK == 0)
    {
        char *filename_config = argv[1];
        printf ("\n");
        printf ("######################################################################\n");
        printf ("CONFIG: %s\n", filename_config);
        flag = read_config(filename_config,
            &xN, &yN, &zN,
            &need_R0, &need_S, &need_H,
            keyword_M, keyword_R0, keyword_S, keyword_H, keyword_Rp,
            method, &beta, &lower_bound, &upper_bound,
            &num_loop, &num_iteration,
            &sigma0, &sigmar, &th);
        if (flag != 0)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fflush(stdout);
    }

    // Broadcacst parameters
    MPI_Bcast(&xN,            1,   MPI_AINT,   0, MPI_COMM_WORLD);
    MPI_Bcast(&yN,            1,   MPI_AINT,   0, MPI_COMM_WORLD);
    MPI_Bcast(&zN,            1,   MPI_AINT,   0, MPI_COMM_WORLD);
    MPI_Bcast(&need_R0,       1,   MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&need_S,        1,   MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&need_H,        1,   MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(keyword_M,      500, MPI_CHAR,   0, MPI_COMM_WORLD);
    MPI_Bcast(keyword_R0,     500, MPI_CHAR,   0, MPI_COMM_WORLD);
    MPI_Bcast(keyword_S,      500, MPI_CHAR,   0, MPI_COMM_WORLD);
    MPI_Bcast(keyword_H,      500, MPI_CHAR,   0, MPI_COMM_WORLD);
    MPI_Bcast(keyword_Rp,     500, MPI_CHAR,   0, MPI_COMM_WORLD);
    MPI_Bcast(method,         500, MPI_CHAR,   0, MPI_COMM_WORLD);
    MPI_Bcast(&beta,          1,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lower_bound,   1,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&upper_bound,   1,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_loop,      1,   MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&num_iteration, 1,   MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&sigma0,        1,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sigmar,        1,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&th,            1,   MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print configurations
    if (RANK == 0)
    {
        printf ("\n");
        printf ("Parameters:\n");
        printf ("\n");
        printf ("    xN:            %ld\n", xN);
        printf ("    yN:            %ld\n", yN);
        printf ("    zN:            %ld\n", zN);
        printf ("\n");
        printf ("    need_R0:       %d\n", need_R0);
        printf ("    need_S:        %d\n", need_S);
        printf ("    need_H:        %d\n", need_H);
        printf ("\n");
        printf ("    keyword M:     %s\n", keyword_M);
        printf ("    keyword R0:    %s\n", keyword_R0);
        printf ("    keyword S:     %s\n", keyword_S);
        printf ("    keyword H:     %s\n", keyword_H);
        printf ("    keyword Rp:    %s\n", keyword_Rp);
        printf ("\n");
        printf ("    method:        %s\n", method);
        printf ("    beta:          %f\n", beta);
        printf ("    lower_bound:   %f\n", lower_bound);
        printf ("    upper_bound:   %f\n", upper_bound);
        printf ("\n");
        printf ("    num_loop:      %d\n", num_loop);
        printf ("    num_iteration: %d\n", num_iteration);
        printf ("\n");
        printf ("    sigma0:        %f\n", sigma0);
        printf ("    sigmar:        %f\n", sigmar);
        printf ("    th:            %f\n", th);
        printf ("######################################################################\n");
        printf ("\n");
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

    if (RANK == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading M", time_str);
        fflush(stdout);
    }

    // Fourier modulus
    double *local_M = (double*)malloc(sizeof(double) * local_indexN);
    flag = read_3d_double(keyword_M, x_start, x_end, yN, zN, local_M);
    if (flag != 0)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (RANK == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading/creating R0", time_str);
        fflush(stdout);
    }

    // Initial model in real space
    double *local_R0 = (double*)malloc(sizeof(double) * local_indexN);
    if (need_R0 == 0)
    {
        create_R0(local_indexN, local_R0);
    }
    else
    {
        flag = read_3d_double(keyword_R0, x_start, x_end, yN, zN, local_R0);
        if (flag != 0)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (RANK == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading/creating S", time_str);
        fflush(stdout);
    }

    // Support in real space
    uint8_t *local_S = (uint8_t*)malloc(sizeof(double) * local_indexN);
    if (need_S == 0)
    {
        create_S(local_indexN, local_S);
    }
    else
    {
        flag = read_3d_uint8(keyword_S, x_start, x_end, yN, zN, local_S);
        if (flag != 0)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (RANK == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start reading/creating H", time_str);
        fflush(stdout);
    }

    // Missing part in Fourier space
    uint8_t *local_H = (uint8_t*)malloc(sizeof(double) * local_indexN);
    if (need_H == 0)
    {
        create_H(local_indexN, local_H);
    }
    else
    {
        flag = read_3d_uint8(keyword_H, x_start, x_end, yN, zN, local_H);
        if (flag != 0)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (RANK == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start creating FFT/IFFT plans", time_str);
        fflush(stdout);
    }

    // Prepare for model's Fourier transform
    // RC: Real space, complex
    // F: Fourier space, complex
    fftw_complex *local_RC, *local_F;
    local_RC = fftw_alloc_complex(local_alloc);
    local_F = fftw_alloc_complex(local_alloc);
    // Make plans for fft and ifft
    fftw_plan plan_fft, plan_ifft;
    plan_fft  = fftw_mpi_plan_dft_3d(xN, yN, zN, local_RC, local_F, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
    plan_ifft = fftw_mpi_plan_dft_3d(xN, yN, zN, local_F, local_RC, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);

    // Prepare for shrink wrap
    // kernel and F_kernel: Gaussian kernel and its Fourier transform
    // R_abs and F_R_abs: model |RC| and its Fourier transform
    // R_conv and F_R_conv: convolution of |RC| and its Fourier transform
    fftw_complex *local_kernel, *local_F_kernel;
    fftw_complex *local_R_abs,  *local_F_R_abs;
    fftw_complex *local_R_conv, *local_F_R_conv;
    local_kernel   = fftw_alloc_complex(local_alloc);
    local_F_kernel = fftw_alloc_complex(local_alloc);
    local_R_abs    = fftw_alloc_complex(local_alloc);
    local_F_R_abs  = fftw_alloc_complex(local_alloc);
    local_R_conv   = fftw_alloc_complex(local_alloc);
    local_F_R_conv = fftw_alloc_complex(local_alloc);
    // Make plans for fft and ifft
    fftw_plan plan_fft_kernel, plan_fft_R_abs, plan_ifft_R_conv;
    plan_fft_kernel  = fftw_mpi_plan_dft_3d(xN, yN, zN, local_kernel, local_F_kernel, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE);
    plan_fft_R_abs   = fftw_mpi_plan_dft_3d(xN, yN, zN, local_R_abs,  local_F_R_abs,  MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE);
    plan_ifft_R_conv = fftw_mpi_plan_dft_3d(xN, yN, zN, local_F_R_conv, local_R_conv, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

    if (RANK == 0)
    {
        printf ("\n");
        get_time_str(time_str);
        printf ("%-50s %s\n", "Optimization starts", time_str);
    }

    // Start loops
    initialize_RC(local_indexN, local_RC, local_R0);
    c_1 = clock();
    run_loop(num_loop, sigma0, sigmar, th,
        method, num_iteration, beta, lower_bound, upper_bound,
        x_start, x_end, xN, yN, zN,
        indexN, local_indexN,
        plan_fft_kernel, plan_fft_R_abs, plan_ifft_R_conv,
        plan_fft, plan_ifft,
        local_RC, local_F,
        local_M, local_S, local_H,
        local_kernel, local_R_abs, local_R_conv,
        local_F_kernel, local_F_R_abs, local_F_R_conv,
        &t_iter_for, &t_iter_fft, &t_wrap_for, &t_wrap_fft);
    //
    c_2 = clock();

    if (RANK == 0)
    {
        printf ("\n");
        get_time_str(time_str);
        printf ("%-50s %s\n", "Start writing Rp", time_str);
        fflush(stdout);
    }

    // Write the reconstruction
    double *local_Rp = (double*)malloc(sizeof(double) * local_indexN);
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_Rp[i] = local_RC[i][0];
    }
    flag = write_3d_double(keyword_Rp, x_start, x_end, yN, zN, local_Rp);
    if (flag != 0)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (RANK == 0)
    {
        get_time_str(time_str);
        printf ("%-50s %s\n", "Task completed", time_str);
        fflush(stdout);
    }

    // Print running time
    t_ini  = ((double)(c_1 - c_0)) / CLOCKS_PER_SEC;
    t_opt = ((double)(c_2 - c_1)) / CLOCKS_PER_SEC;
    t_iter_for = t_iter_for / CLOCKS_PER_SEC;
    t_iter_fft = t_iter_fft / CLOCKS_PER_SEC;
    t_wrap_for = t_wrap_for / CLOCKS_PER_SEC;
    t_wrap_fft = t_wrap_fft / CLOCKS_PER_SEC;

    if (RANK == 0)
    {
        printf ("\n");
        printf ("######################################################################\n");
        printf ("Summary\n");
        printf ("\n");
        printf ("    xN:                             %ld\n", xN);
        printf ("    yN:                             %ld\n", yN);
        printf ("    zN:                             %ld\n", zN);
        printf ("    method:                         %s\n", method);
        printf ("    num_loop:                       %d\n", num_loop);
        printf ("    num_iteration:                  %d\n", num_iteration);
        printf ("    Number of processes:            %d\n", SIZE);
        printf ("    Runtime for initialization (s): %f\n", t_ini);
        printf ("    Runtime for optimization (s):   %f\n", t_opt);
        printf ("\n");
        printf ("    ------------------------------------------------------------------\n");
        printf ("    Runtime for Iter-FOR (s):       %f\n", t_iter_for);
        printf ("    Runtime for Iter-FFT (s):       %f\n", t_iter_fft);
        printf ("    Runtime for SW-FOR (s):         %f\n", t_wrap_for);
        printf ("    Runtime for SW-FFT (s):         %f\n", t_wrap_fft);
        printf ("######################################################################\n");
        printf ("\n");
        fclose(file_log);
    }

    // Clean and exit
    free(time_str);

    free(keyword_M);
    free(keyword_R0);
    free(keyword_S);
    free(keyword_H);
    free(keyword_Rp);
    free(method);

    free(local_M);
    free(local_R0);
    free(local_S);
    free(local_H);
    free(local_Rp);

    free(local_RC);
    free(local_F);

    free(local_kernel);
    free(local_F_kernel);
    free(local_R_abs);
    free(local_F_R_abs);
    free(local_R_conv);
    free(local_F_R_conv);

    fftw_destroy_plan(plan_fft);
    fftw_destroy_plan(plan_ifft);
    fftw_destroy_plan(plan_fft_kernel);
    fftw_destroy_plan(plan_fft_R_abs);
    fftw_destroy_plan(plan_ifft_R_conv);

    fftw_mpi_cleanup();
    MPI_Finalize();

    return 0;
}


// By default, the initial model consists of random numbers across the entire volume
int create_R0(ptrdiff_t local_indexN, double *local_R0)
{
    srand(time(NULL));
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        // Value range [0, 1)
        local_R0[i] = (double)rand() / RAND_MAX;
    }
    return 0;
}


// By default, the support extends across the entire volume
int create_S(ptrdiff_t local_indexN, uint8_t *local_S)
{
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_S[i] = 1;
    }
    return 0;
}


// By default, there is no missing part in Fourier space
int create_H(ptrdiff_t local_indexN, uint8_t *local_H)
{
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_H[i] = 0;
    }
    return 0;
}


// Copy local_R to the real part of local_RC
int initialize_RC(ptrdiff_t local_indexN, fftw_complex *local_RC, double *local_R0)
{
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC[i][0] = local_R0[i];
        local_RC[i][1] = 0.0;
    }
    return 0;
}


// Run iterations without shrink wrap
int run_iteration(fftw_plan plan_fft, fftw_plan plan_ifft,
    char *method, int num_iteration, double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H,
    double *t_iter_for, double *t_iter_fft)
{
    int (*optFUN[])(fftw_plan, fftw_plan,
        double, double, double,
        ptrdiff_t, ptrdiff_t,
        fftw_complex*, fftw_complex*,
        double*, uint8_t*, uint8_t*,
        double*, double*) = {optHIO, optER, optASR, optHPR, optRAAR, optDM};

    int choice;
    if (strcmp(method, "HIO") == 0)       choice = 0;
    else if (strcmp(method, "ER") == 0)   choice = 1;
    else if (strcmp(method, "ASR") == 0)  choice = 2;
    else if (strcmp(method, "HPR") == 0)  choice = 3;
    else if (strcmp(method, "RAAR") == 0) choice = 4;
    else if (strcmp(method, "DM") == 0)   choice = 5;

    char* time_str = (char*)malloc(100 * sizeof(char));
    for (int iter = 0; iter < num_iteration; iter++)
    {
        if (RANK == 0)
        {
            get_time_str(time_str);
            printf ("%-19s %-30d %s\n", "    iteration", iter, time_str);
            fflush(stdout);
        }

        optFUN[choice](plan_fft, plan_ifft,
            beta, lower_bound, upper_bound,
            indexN, local_indexN,
            local_RC, local_F,
            local_M, local_S, local_H,
            t_iter_for, t_iter_fft);
    }
    free(time_str);

    return 0;
}


// Run shrink wrap loops
int run_loop(int num_loop, double sigma0, double sigmar, double th,
    char *method, int num_iteration, double beta, double lower_bound, double upper_bound,
    ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t xN, ptrdiff_t yN, ptrdiff_t zN,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_plan plan_fft_kernel, fftw_plan plan_fft_R_abs, fftw_plan plan_ifft_R_conv,
    fftw_plan plan_fft, fftw_plan plan_ifft,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H,
    fftw_complex *local_kernel, fftw_complex *local_R_abs, fftw_complex *local_R_conv,
    fftw_complex *local_F_kernel, fftw_complex *local_F_R_abs, fftw_complex *local_F_R_conv,
    double *t_iter_for, double *t_iter_fft, double *t_wrap_for, double *t_wrap_fft)
{
    double sigma;
    char* time_str = (char*)malloc(100 * sizeof(char));
    for (int loop = 0; loop < num_loop; loop++)
    {
        if (RANK == 0)
        {
            printf ("\n");
            get_time_str(time_str);
            printf ("%-4s %-45d %s\n", "Loop", loop, time_str);
            fflush(stdout);
        }

        run_iteration(plan_fft, plan_ifft,
            method, num_iteration, beta, lower_bound, upper_bound,
            indexN, local_indexN,
            local_RC, local_F,
            local_M, local_S, local_H,
            t_iter_for, t_iter_fft);
        // Current RC is real, give imaginary part by projM()
        projM(plan_fft, plan_ifft,
            indexN, local_indexN,
            local_RC, local_F, local_M, local_H,
            t_wrap_for, t_wrap_fft);
        sigma = sigma0 * pow(1.0 - sigmar, loop);
        get_kernel(x_start, x_end, xN, yN, zN, sigma, local_kernel, t_wrap_for);
        get_R_conv(indexN, local_indexN, local_RC,
            plan_fft_kernel, plan_fft_R_abs, plan_ifft_R_conv,
            local_R_abs, local_R_conv,
            local_F_kernel, local_F_R_abs, local_F_R_conv, t_wrap_for, t_wrap_fft);
        update_S(th, local_indexN, local_S, local_R_conv, MPI_COMM_WORLD, t_wrap_for);
        // Cast away imaginary part of RC
        for (ptrdiff_t i = 0; i < local_indexN; i++)
        {
            local_RC[i][1] = 0.0;
        }
    }
    free(time_str);

    return 0;
}
