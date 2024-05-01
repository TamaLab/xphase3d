#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "hdf5.h"
#include "fileio.h"


// Obtain current time
void get_time_str(char* time_str)
{
    time_t current_time;
    struct tm *time_info;
    time(&current_time);
    time_info = localtime(&current_time);
    strftime(time_str, 100, "%H:%M:%S %d/%m/%Y", time_info);
}


// Read a 2D slice, datatype double / H5T_IEEE_F64LE
int read_2d_double(char *filename, ptrdiff_t yN, ptrdiff_t zN, double *slice)
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;
    hid_t dtype;
    H5T_class_t class;
    H5T_order_t order;
    size_t size;
    int ndims;

    // Turn off verbose error message of hdf5
    status = H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    // Open the file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
    {
        printf ("Error in reading %s:\n", filename);
        printf ("    cannot find or open the file\n");
        fflush(stdout);
        return 1;
    }

    // Open the dataset
    dataset_id = H5Dopen(file_id, "/dataset", H5P_DEFAULT);
    if (dataset_id < 0)
    {
        printf ("Error in reading %s:\n", filename);
        printf ("    cannot open /dataset\n");
        fflush(stdout);
        status = H5Fclose(file_id);
        return 1;
    }

    // Verify the dataset's datatype
    dtype = H5Dget_type(dataset_id);
    class = H5Tget_class(dtype);
    order = H5Tget_order(dtype);
    size = H5Tget_size(dtype);
    if (class != H5T_FLOAT || order != H5T_ORDER_LE || size != 8)
    {
        printf ("Error in reading %s:\n", filename);
        printf ("    require datatype float, 8 bytes, little-endian\n");
        fflush(stdout);
        status = H5Dclose(dataset_id);
        status = H5Fclose(file_id);
        return 1;
    }

    // Verify the dataset's dimensions
    dataspace_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_ndims(dataspace_id);
    hsize_t dims[ndims];
    status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    if (ndims != 2 || dims[0] != (hsize_t)yN || dims[1] != (hsize_t)zN)
    {
        printf ("Error in reading %s:\n", filename);
        printf ("    require dimensions %ld * %ld\n", yN, zN);
        fflush(stdout);
        status = H5Sclose(dataspace_id);
        status = H5Dclose(dataset_id);
        status = H5Fclose(file_id);
        return 1;
    }

    // Read the dataset
    status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, slice);
    if (status < 0)
    {
        printf ("Error in reading %s:\n", filename);
        printf ("    failed in reading the dataset");
        fflush(stdout);
        status = H5Sclose(dataspace_id);
        status = H5Dclose(dataset_id);
        status = H5Fclose(file_id);
        return 1;
    }

    status = H5Sclose(dataspace_id);
    status = H5Dclose(dataset_id);
    status = H5Fclose(file_id);
    return 0;
}


// Read a 2D slice, datatype uint8 / H5T_NATIVE_CHAR
int read_2d_uint8(char *filename, ptrdiff_t yN, ptrdiff_t zN, uint8_t *slice)
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;
    hid_t dtype;
    H5T_class_t class;
    size_t size;
    int ndims;

    // Turn off verbose error message of hdf5
    status = H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    // Open the file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
    {
        printf ("Error in reading %s:\n", filename);
        printf ("    cannot find or open the file\n");
        fflush(stdout);
        return 1;
    }

    // Open the dataset
    dataset_id = H5Dopen(file_id, "/dataset", H5P_DEFAULT);
    if (dataset_id < 0)
    {
        printf ("Error in reading %s:\n", filename);
        printf ("    cannot open /dataset\n");
        fflush(stdout);
        status = H5Fclose(file_id);
        return 1;
    }

    // Verify the dataset's datatype
    dtype = H5Dget_type(dataset_id);
    class = H5Tget_class(dtype);
    size = H5Tget_size(dtype);
    if (class != H5T_INTEGER || size != 1)
    {
        printf ("Error in reading %s:\n", filename);
        printf ("    require datatype int, 1 byte\n");
        fflush(stdout);
        status = H5Dclose(dataset_id);
        status = H5Fclose(file_id);
        return 1;
    }

    // Verify the dataset's dimensions
    dataspace_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_ndims(dataspace_id);
    hsize_t dims[ndims];
    status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    if (ndims != 2 || dims[0] != (hsize_t)yN || dims[1] != (hsize_t)zN)
    {
        printf ("Error in reading %s:\n", filename);
        printf ("    require dimensions %ld * %ld\n", yN, zN);
        fflush(stdout);
        status = H5Sclose(dataspace_id);
        status = H5Dclose(dataset_id);
        status = H5Fclose(file_id);
        return 1;
    }

    // Read the dataset
    status = H5Dread(dataset_id, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, slice);
    if (status < 0)
    {
        printf ("Error in reading %s:\n", filename);
        printf ("    failed in reading the dataset");
        fflush(stdout);
        status = H5Sclose(dataspace_id);
        status = H5Dclose(dataset_id);
        status = H5Fclose(file_id);
        return 1;
    }

    status = H5Sclose(dataspace_id);
    status = H5Dclose(dataset_id);
    status = H5Fclose(file_id);
    return 0;
}


// Read a 3D block, datatype double / H5T_IEEE_F64LE
int read_3d_double(char *keyword, ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t yN, ptrdiff_t zN, double *block)
{
    double *slice;
    char *filename = (char*)malloc(sizeof(char) * 500);
    // Read slices one by one
    int flag = 0;
    for (ptrdiff_t x = x_start; x < x_end; x++)
    {
        sprintf (filename, "%s_%ld.h5", keyword, x);
        slice = block + (x - x_start) * yN * zN;
        flag = flag + read_2d_double(filename, yN, zN, slice);
    }

    free(filename);
    return flag;
}


// Read a 3D block, datatype uint8 / H5T_NATIVE_CHAR
int read_3d_uint8(char *keyword, ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t yN, ptrdiff_t zN, uint8_t *block)
{
    uint8_t *slice;
    char *filename = (char*)malloc(sizeof(char) * 500);
    // Read slices one by one
    int flag = 0;
    for (ptrdiff_t x = x_start; x < x_end; x++)
    {
        sprintf (filename, "%s_%ld.h5", keyword, x);
        slice = block + (x - x_start) * yN * zN;
        flag = flag + read_2d_uint8(filename, yN, zN, slice);
    }

    free(filename);
    return flag;
}


// Write a 2D slice, datatype double / H5T_IEEE_F64LE
int write_2d_double(char *filename, ptrdiff_t yN, ptrdiff_t zN, double *slice)
{
    hsize_t dims[] = {yN, zN};
    hid_t file_id, dataspace_id, dataset_id;
    herr_t status;

    // Turn off verbose error message of hdf5
    status = H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
    {
        printf ("Error in writing %s:\n", filename);
        printf ("    cannot find or open the file\n");
        fflush(stdout);
        return 1;
    }

    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/dataset", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, slice);
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);
    status = H5Fclose(file_id);
    status += 0;
    return 0;
}


// Write a 2D slice, datatype uint8 / H5T_NATIVE_CHAR
int write_2d_uint8(char *filename, ptrdiff_t yN, ptrdiff_t zN, uint8_t *slice)
{
    hsize_t dims[] = {yN, zN};
    hid_t file_id, dataspace_id, dataset_id;
    herr_t status;

    // Turn off verbose error message of hdf5
    status = H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
    {
        printf ("Error in writing %s:\n", filename);
        printf ("    cannot find or open the file\n");
        fflush(stdout);
        return 1;
    }

    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/dataset", H5T_NATIVE_CHAR, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, slice);
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);
    status = H5Fclose(file_id);
    status += 0;
    return 0;
}


// Write a 3D block, datatype double / H5T_IEEE_F64LE
int write_3d_double(char *keyword, ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t yN, ptrdiff_t zN, double *block)
{
    double *slice;
    char *filename = (char*)malloc(sizeof(char) * 500);
    // Write slices one by one
    int flag = 0;
    for (ptrdiff_t x = x_start; x < x_end; x++)
    {
        sprintf (filename, "%s_%ld.h5", keyword, x);
        slice = block + (x - x_start) * yN * zN;
        flag = flag + write_2d_double(filename, yN, zN, slice);
    }

    free(filename);
    return flag;
}


// Write a 3D block, datatype uint8 / H5T_NATIVE_CHAR
int write_3d_uint8(char *keyword, ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t yN, ptrdiff_t zN, uint8_t *block)
{
    uint8_t *slice;
    char *filename = (char*)malloc(sizeof(char) * 500);
    // Write slices one by one
    int flag = 0;
    for (ptrdiff_t x = x_start; x < x_end; x++)
    {
        sprintf (filename, "%s_%ld.h5", keyword, x);
        slice = block + (x - x_start) * yN * zN;
        flag = flag + write_2d_uint8(filename, yN, zN, slice);
    }

    free(filename);
    return flag;
}


// Write a binned 3D array in one file, datatype double / H5T_IEEE_F64LE
int write_3d_double_in_one(char *filename, ptrdiff_t xN, ptrdiff_t yN, ptrdiff_t zN, double *array)
{
    hsize_t dims[] = {xN, yN, zN};
    hid_t file_id, dataspace_id, dataset_id;
    herr_t status;

    // Turn off verbose error message of hdf5
    status = H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
    {
        printf ("Error in writing %s:\n", filename);
        printf ("    cannot find or open the file\n");
        fflush(stdout);
        return 1;
    }

    dataspace_id = H5Screate_simple(3, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/dataset", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);
    status = H5Fclose(file_id);
    status += 0;
    return 0;
}


// Write 1D data (FSC or PRTF) in plain text
int write_profile(char *filename, ptrdiff_t kN, double *profile)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf ("Error in writing %s:\n", filename);
        printf ("    cannot find or open the file");
        fflush(stdout);
        return 1;
    }

    int flag;
    for (ptrdiff_t i = 0; i < kN; i++)
    {
        flag = fprintf(file, "%-10ld\t%lf\n", i, profile[i]);
        if (flag < 0)
        {
            printf ("Error in writing %s:\n", filename);
            printf ("    failed in writing data");
            fflush(stdout);
            return 1;
        }
    }

    return 0;
}


// str2: the '#' in str1 is replaced by number; used in merge.c and prtf.c
void replace(char *str1, char *str2, ptrdiff_t number)
{
    sprintf(str2, "%s", str1);
    char *ptr = strchr(str2, '#');
    if (ptr != NULL)
    {
        char str_number[25];
        sprintf(str_number, "%ld", number);
        size_t len_number = strlen(str_number);
        size_t len_right = strlen(ptr + 1);
        memmove(ptr + len_number, ptr + 1, len_right + 1);
        memcpy(ptr, str_number, len_number);
    }
}


// Remove leading and trailing spaces and tabs from a string.; used in read_config()
void trim(char string[])
{
    if (string == NULL)
    {
        return;
    }

    int length = strlen(string);

    // Trim leading spaces and tabs
    int start = 0;
    while (string[start] == '\t' || string[start] == ' ')
    {
        start++;
    }

    // Trim trailing spaces and tabs
    int end = length - 1;
    while (string[end] == '\t' || string[end] == ' ')
    {
        end--;
    }

    // Adjust the string in place
    if (start <= end)
    {
        memmove(string, string + start, end - start + 1);
        string[end - start + 1] = '\0';
    }
    else
    {
        string[0] = '\0';
    }
}


// Read configurations
int read_config(char *filename_config,
    ptrdiff_t *xN, ptrdiff_t *yN, ptrdiff_t *zN,
    int *need_R0, int *need_S, int *need_H,
    char *keyword_M, char *keyword_R0, char *keyword_S, char *keyword_H, char *keyword_Rp,
    char *method, double *beta, double *lower_bound, double *upper_bound,
    int *num_loop, int *num_iteration,
    double *sigma0, double *sigmar, double *th)
{
    int flag = 0;

    FILE *file;
    file = fopen(filename_config, "r");
    if (file == NULL)
    {
        printf ("Error in reading %s:\n", filename_config);
        printf ("    cannot find or open the file\n");
        fflush(stdout);
        flag = 1;
        return flag;
    }

    // Check if any parameter will be omitted
    int xN_r = 0; int yN_r = 0; int zN_r = 0;
    int nR0_r = 0; int nS_r = 0; int nH_r = 0;
    int keyM_r = 0; int keyR0_r = 0; int keyS_r = 0; int keyH_r = 0; int keyRp_r = 0;
    int method_r = 0; int beta_r = 0; int lbound_r = 0; int ubound_r = 0;
    int nloop_r = 0; int niter_r = 0;
    int sigma0_r = 0; int sigmar_r = 0; int th_r = 0;

    // Available methods for phase retrieval
    char methods[][5] = {"HIO", "ER", "ASR", "HPR", "RAAR", "DM"};
    int num_methods = 6;
    int method_flag;

    // Read the text line by line
    // The maximum length of one line is 1000 characters
    char line[1000], par_name[500], par_value[500];
    int onechar;
    char *endptr;

    while (fgets(line, sizeof(line), file) != NULL)
    {
        // If one line exceeds the maximum length, discard the remaining characters in this line
        if (line[strlen(line) - 1] != '\n' && feof(file) == 0)
        {
            onechar = fgetc(file);
            while (onechar != '\n' && onechar != EOF)
            {
                onechar = fgetc(file);
            }
        }

        // Remove leading and trailing spaces and tabs
        trim(line);

        // Skip a line that starts with '#'
        if (line[0] == '#')
        {
            continue;
        }

        // Split the line by ':'
        if (sscanf(line, "%499[^:]:%499[^\n]", par_name, par_value) == 2)
        {
            trim(par_name);
            trim(par_value);

            // xN
            if (strcmp(par_name, "XN") == 0)
            {
                xN_r++;
                *xN = strtol(par_value, &endptr, 10);
                if (*endptr != '\0')
                {
                    printf ("    - XN is not an integer\n");
                    flag = 1;
                }
            }

            // yN
            else if (strcmp(par_name, "YN") == 0)
            {
                yN_r++;
                *yN = strtol(par_value, &endptr, 10);
                if (*endptr != '\0')
                {
                    printf ("    - YN is not an integer\n");
                    flag = 1;
                }
            }

            // zN
            else if (strcmp(par_name, "ZN") == 0)
            {
                zN_r++;
                *zN = strtol(par_value, &endptr, 10);
                if (*endptr != '\0')
                {
                    printf ("    - ZN is not an integer\n");
                    flag = 1;
                }
            }

            // need_R0
            else if (strcmp(par_name, "NEED R0") == 0)
            {
                nR0_r++;
                *need_R0 = strtol(par_value, &endptr, 10);
                if (*endptr != '\0' || (*need_R0 != 0 && *need_R0 != 1))
                {
                    printf("    - NEED R0 should be either 0 or 1\n");
                    flag = 1;
                }
            }

            // need_S
            else if (strcmp(par_name, "NEED S") == 0)
            {
                nS_r++;
                *need_S = strtol(par_value, &endptr, 10);
                if (*endptr != '\0' || (*need_S != 0 && *need_S != 1))
                {
                    printf("    - NEED S should be either 0 or 1\n");
                    flag = 1;
                }
            }

            // need_H
            else if (strcmp(par_name, "NEED H") == 0)
            {
                nH_r++;
                *need_H = strtol(par_value, &endptr, 10);
                if (*endptr != '\0' || (*need_H != 0 && *need_H != 1))
                {
                    printf("    - NEED H should be either 0 or 1\n");
                    flag = 1;
                }
            }

            // keyword_M
            else if (strcmp(par_name, "KEYWORD M") == 0)
            {
                keyM_r++;
                strcpy(keyword_M, par_value);
            }

            // keyword_R0
            else if (strcmp(par_name, "KEYWORD R0") == 0)
            {
                keyR0_r++;
                strcpy(keyword_R0, par_value);
            }

            // keyword_S
            else if (strcmp(par_name, "KEYWORD S") == 0)
            {
                keyS_r++;
                strcpy(keyword_S, par_value);
            }

            // keyword_H
            else if (strcmp(par_name, "KEYWORD H") == 0)
            {
                keyH_r++;
                strcpy(keyword_H, par_value);
            }

            // keyword_Rp
            else if (strcmp(par_name, "KEYWORD Rp") == 0)
            {
                keyRp_r++;
                strcpy(keyword_Rp, par_value);
            }

            // method
            else if (strcmp(par_name, "METHOD") == 0)
            {
                method_r++;
                strcpy(method, par_value);
                method_flag = 1;
                for (int i = 0; i < num_methods; i++)
                {
                    if (strcmp(method, methods[i]) == 0)
                    {
                        method_flag = 0;
                    }
                }
                if (method_flag == 1)
                {
                    printf ("    - METHOD %s is not available\n", method);
                    printf ("    - Available methods:");
                    for (int i = 0; i < num_methods; i++)
                    {
                        printf (" %s", methods[i]);
                    }
                    printf ("\n");
                    flag = 1;
                }
            }

            // beta
            else if (strcmp(par_name, "BETA") == 0)
            {
                beta_r++;
                *beta = strtod(par_value, &endptr);
                if (*endptr != '\0')
                {
                    printf ("    - BETA is not a float number\n");
                    flag = 1;
                }
            }

            // lower_bound
            else if (strcmp(par_name, "LOWER BOUND") == 0)
            {
                lbound_r++;
                *lower_bound = strtod(par_value, &endptr);
                if (*endptr != '\0')
                {
                    printf ("    - LOWER BOUND is not a float number\n");
                    flag = 1;
                }
            }

            // upper_bound
            else if (strcmp(par_name, "UPPER BOUND") == 0)
            {
                ubound_r++;
                *upper_bound = strtod(par_value, &endptr);
                if (*endptr != '\0')
                {
                    printf ("    - UPPER BOUND is not a float number\n");
                    flag = 1;
                }
            }

            // loopN
            else if (strcmp(par_name, "NUM LOOP") == 0)
            {
                nloop_r++;
                *num_loop = strtol(par_value, &endptr, 10);
                if (*endptr != '\0')
                {
                    printf ("    - NUM LOOP is not an integer\n");
                    flag = 1;
                }
            }

            // sloopN
            else if (strcmp(par_name, "NUM ITERATION") == 0)
            {
                niter_r++;
                *num_iteration = strtol(par_value, &endptr, 10);
                if (*endptr != '\0')
                {
                    printf ("    - NUM ITERATION is not an integer\n");
                    flag = 1;
                }
            }

            // sigma0
            else if (strcmp(par_name, "SIGMA0") == 0)
            {
                sigma0_r++;
                *sigma0 = strtod(par_value, &endptr);
                if (*endptr != '\0')
                {
                    printf ("    - SIGMA0 is not a float number\n");
                    flag = 1;
                }
            }

            // sigmar
            else if (strcmp(par_name, "SIGMAR") == 0)
            {
                sigmar_r++;
                *sigmar = strtod(par_value, &endptr);
                if (*endptr != '\0')
                {
                    printf ("    - SIGMAR is not a float number\n");
                    flag = 1;
                }
            }

            // th
            else if (strcmp(par_name, "TH") == 0)
            {
                th_r++;
                *th = strtod(par_value, &endptr);
                if (*endptr != '\0')
                {
                    printf ("    - TH is not a float number\n");
                    flag = 1;
                }
            }
        }
    }

    // Check if any parameter is omitted
    if (xN_r     == 0)    {printf ("    - XN is not defined\n");            flag = 1;}
    if (yN_r     == 0)    {printf ("    - YN is not defined\n");            flag = 1;}
    if (zN_r     == 0)    {printf ("    - ZN is not defined\n");            flag = 1;}
    if (nR0_r    == 0)    {printf ("    - NEED R0 is not defined\n");       flag = 1;}
    if (nS_r     == 0)    {printf ("    - NEED S is not defined\n");        flag = 1;}
    if (nH_r     == 0)    {printf ("    - NEED H is not defined\n");        flag = 1;}
    if (keyM_r   == 0)    {printf ("    - KEYWORD M is not defined\n");     flag = 1;}
    if (keyRp_r  == 0)    {printf ("    - KEYWORD Rp is not defined\n");    flag = 1;}
    if (method_r == 0)    {printf ("    - METHOD is not defined\n");        flag = 1;}
    if (beta_r   == 0)    {printf ("    - BETA is not defined\n");          flag = 1;}
    if (lbound_r == 0)    {printf ("    - LOWER BOUND is not defined\n");   flag = 1;}
    if (ubound_r == 0)    {printf ("    - UPPER BOUND is not defined\n");   flag = 1;}
    if (nloop_r  == 0)    {printf ("    - NUM LOOP is not defined\n");      flag = 1;}
    if (niter_r == 0)     {printf ("    - NUM ITERATION is not defined\n"); flag = 1;}
    if (sigma0_r == 0)    {printf ("    - SIGMA0 is not defined\n");        flag = 1;}
    if (sigmar_r == 0)    {printf ("    - SIGMAR is not defined\n");        flag = 1;}
    if (th_r     == 0)    {printf ("    - TH is not defined\n");            flag = 1;}

    if (keyR0_r  == 0 && *need_R0 != 0)    {printf ("    - KEYWORD R0 is not defined\n");  flag = 1;}
    if (keyS_r   == 0 && *need_S  != 0)    {printf ("    - KEYWORD S is not defined\n");   flag = 1;}
    if (keyH_r   == 0 && *need_H  != 0)    {printf ("    - KEYWORD H is not defined\n");   flag = 1;}

    if (flag == 1)
    {
        printf ("Error in reading parameters from %s\n", filename_config);
    }

    fflush(stdout);
    return flag;
}
