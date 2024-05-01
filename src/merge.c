#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fftw3-mpi.h"
#include "fileio.h"


// To merge a series of aligned 3D density models


void print_usage()
{
    printf ("\n");
    printf ("Usage: mpiexec -n <num_processes> merge IN START END OUT XN YN ZN\n\n");
    printf ("Arguments\n");
    printf ("    - mpiexec -n <num_processes> : Number of MPI processes\n");
    printf ("    - IN                         : Keyword of input 3D density models\n");
    printf ("                                   Use '#' as wildcard for sequential number\n");
    printf ("    - START                      : Starting sequential number\n");
    printf ("    - END                        : Ending sequential number\n");
    printf ("    - OUT                        : Keyword of the output merged 3D density model\n");
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
    char *keyword_out = argv[4];
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
    char filename_log[500];
    strcpy(filename_log, keyword_out);
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
        printf ("XPHASE3D - MERGE\n");
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

    // Merge slices (start to end)
    double *slice = (double*)malloc(sizeof(double) * yN * zN);
    double *SLICE = (double*)malloc(sizeof(double) * yN * zN);
    char *keyword_in_s = (char*)malloc(sizeof(char) * 500);
    char *filename_in  = (char*)malloc(sizeof(char) * 500);
    char *filename_out = (char*)malloc(sizeof(char) * 500);

    int flag = 0;
    for (ptrdiff_t x = x_start; x < x_end; x++)
    {
        for (ptrdiff_t i = 0; i < yN * zN; i++)
        {
            SLICE[i] = 0.0;
        }

        for (ptrdiff_t s = start; s < end + 1; s++)
        {
            replace(keyword_in, keyword_in_s, s);
            sprintf(filename_in, "%s_%ld.h5", keyword_in_s, x);
            flag = read_2d_double(filename_in, yN, zN, slice);
            if (flag != 0)
            {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            for (ptrdiff_t i = 0; i < yN * zN; i++)
            {
                SLICE[i] += slice[i];
            }
        }

        for (ptrdiff_t i = 0; i < yN * zN; i++)
        {
            SLICE[i] = SLICE[i] / (double)(end - start + 1);
        }

        sprintf(filename_out, "%s_%ld.h5", keyword_out, x);
        flag = write_2d_double(filename_out, yN, zN, SLICE);
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
    free(slice);
    free(SLICE);
    free(keyword_in_s);
    free(filename_in);
    free(filename_out);

    fftw_mpi_cleanup();
    MPI_Finalize();

    return 0;
}
