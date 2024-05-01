#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fileio.h"


// To bin a large 3D array (uint8) into a smaller one (double) for preview


void print_usage()
{
    printf ("\n");
    printf ("Usage: bin_b BF IN OUT XN YN ZN\n\n");
    printf ("Arguments\n");
    printf ("    - BF                         : Binning factor\n");
    printf ("    - IN                         : Keyword of input file: Original density model\n");
    printf ("    - OUT                        : Name of output file: Binned density model\n");
    printf ("    - XN                         : Size of the X dimension\n");
    printf ("    - YN                         : Size of the Y dimension\n");
    printf ("    - ZN                         : Size of the Z dimension\n");
    printf ("\n");
}


int main(int argc, char **argv)
{

    if (argc != 7)
    {
        print_usage();
        return 1;
    }

    char *keyword_R = argv[2];
    char *filename_R2 = argv[3];

    ptrdiff_t bf, xN, yN, zN;
    char *endptr_bf, *endptr_xN, *endptr_yN, *endptr_zN;
    bf = strtol(argv[1], &endptr_bf, 10);
    xN = strtol(argv[4], &endptr_xN, 10);
    yN = strtol(argv[5], &endptr_yN, 10);
    zN = strtol(argv[6], &endptr_zN, 10);
    if (*endptr_bf != '\0' || *endptr_xN != '\0' || *endptr_yN != '\0' || *endptr_zN != '\0')
    {
        printf ("\n");
        if (*endptr_bf != '\0')    printf ("Error: BF <%s> is not a valid integer\n", argv[1]);
        if (*endptr_xN != '\0')    printf ("Error: XN <%s> is not a valid integer\n", argv[4]);
        if (*endptr_yN != '\0')    printf ("Error: YN <%s> is not a valid integer\n", argv[5]);
        if (*endptr_zN != '\0')    printf ("Error: ZN <%s> is not a valid integer\n", argv[6]);
        print_usage();
        return 1;
    }
    if (bf <= 0)
    {
        printf ("\n");
        printf ("Error: BF must be positive\n");
        print_usage();
        return 1;
    }

    // Create a log file
    FILE *file_log;
    char *time_str = (char*)malloc(100 * sizeof(char));
    char filename_log[500];
    strcpy(filename_log, filename_R2);
    strcat(filename_log, ".log");
    file_log = fopen(filename_log, "a");
    if (file_log == NULL)
    {
        printf ("Error in creating %s\n", filename_log);
        return 1;
    }
    freopen(filename_log, "a", stdout);
    printf ("XPHASE3D - BIN_B\n");
    printf ("\n");
    get_time_str(time_str);
    printf ("%-50s %s\n", "Program started", time_str);
    printf ("\n");
    fflush(stdout);

    // Size of dimensions after binning
    ptrdiff_t XN, YN, ZN;
    XN = xN / bf;
    YN = yN / bf;
    ZN = zN / bf;

    // R1: First step, bin on Y-axis and Z-axis
    double *R1 = (double*)malloc(sizeof(double) * xN * YN * ZN);
    for (int i = 0; i < xN * YN * ZN; i++)
    {
        R1[i] = 0.0;
    }
    int flag;
    char *filename_R = (char*)malloc(sizeof(char) * 500);
    uint8_t *slice_R = (uint8_t*)malloc(sizeof(uint8_t) * yN * zN);
    ptrdiff_t x, y, z;
    ptrdiff_t X, Y, Z;
    ptrdiff_t dx, dy, dz;
    ptrdiff_t INDEX1, index;

    get_time_str(time_str);
    printf ("%-50s %s\n", "Start binning along Y and Z axes", time_str);
    fflush(stdout);

    for (x = 0; x < xN; x++)
    {
        get_time_str(time_str);
        printf ("%s %-35ld %s\n", "    Slice NO. ", x, time_str);
        fflush(stdout);

        sprintf (filename_R, "%s_%ld.h5", keyword_R, x);
        flag = read_2d_uint8(filename_R, yN, zN, slice_R);
        if (flag != 0)
        {
            return 1;
        }
        for (Y = 0; Y < YN; Y++)
        {
            for (Z = 0; Z < ZN; Z++)
            {
                for (dy = 0; dy < bf; dy++)
                {

                    for (dz = 0; dz < bf; dz++)
                    {
                        y = Y * bf + dy;
                        z = Z * bf + dz;
                        INDEX1 = x * YN * ZN + Y * ZN + Z;
                        index = y * zN + z;
                        R1[INDEX1] += slice_R[index];
                    }
                }
            }
        }
    }

    printf ("\n");
    get_time_str(time_str);
    printf ("%-50s %s\n", "Start binning along X axis", time_str);
    printf ("\n");
    fflush(stdout);

    // R2: Second step, bin on X-axis
    double *R2 = (double*)malloc(sizeof(double) * XN * YN * ZN);
    for (int i = 0; i < XN * YN * ZN; i++)
    {
        R2[i] = 0.0;
    }
    ptrdiff_t INDEX2;
    for (X = 0; X < XN; X++)
    {
        for (dx = 0; dx < bf; dx++)
        {
            for (Y = 0; Y < YN; Y++)
            {
                for (Z = 0; Z < ZN; Z++)
                {
                    x = X * bf + dx;
                    INDEX1 = x * YN * ZN + Y * ZN + Z;
                    INDEX2 = X * YN * ZN + Y * ZN + Z;
                    R2[INDEX2] += R1[INDEX1];
                }
            }
        }
    }
    for (ptrdiff_t i = 0; i < XN * YN * ZN; i++)
    {
        R2[i] = R2[i] / (double)(bf * bf * bf);
    }

    get_time_str(time_str);
    printf ("%-50s %s\n", "Start writing OUT", time_str);
    fflush(stdout);

    flag = write_3d_double_in_one(filename_R2, XN, YN, ZN, R2);
    if (flag != 0)
    {
        return 1;
    }

    get_time_str(time_str);
    printf ("%-50s %s\n", "Task completed", time_str);
    printf ("\n");
    fflush(stdout);
    fclose(file_log);

    free(time_str);
    free(R1);
    free(R2);
    free(filename_R);
    free(slice_R);

    return 0;
}
