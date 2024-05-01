#ifndef FILEIO_H
#define FILEIO_H
#include <stddef.h>
#include <stdint.h>

// Obtain current time
void get_time_str(char* time_str);

// Read a 2D slice, datatype double / H5T_IEEE_F64LE
int read_2d_double(char *filename, ptrdiff_t yN, ptrdiff_t zN, double *slice);

// Read a 2D slice, datatype uint8 / H5T_NATIVE_CHAR
int read_2d_uint8(char *filename, ptrdiff_t yN, ptrdiff_t zN, uint8_t *slice);

// Read a 3D block, datatype double / H5T_IEEE_F64LE
int read_3d_double(char *keyword, ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t yN, ptrdiff_t zN, double *block);

// Read a 3D block, datatype uint8 / H5T_NATIVE_CHAR
int read_3d_uint8(char *keyword, ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t yN, ptrdiff_t zN, uint8_t *block);

// Write a 2D slice, datatype double / H5T_IEEE_F64LE
int write_2d_double(char *filename, ptrdiff_t yN, ptrdiff_t zN, double *slice);

// Write a 2D slice, datatype uint8 / H5T_NATIVE_CHAR
int write_2d_uint8(char *filename, ptrdiff_t yN, ptrdiff_t zN, uint8_t *slice);

// Write a 3D block, datatype double / H5T_IEEE_F64LE
int write_3d_double(char *keyword, ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t yN, ptrdiff_t zN, double *block);

// Write a 3D block, datatype uint8 / H5T_NATIVE_CHAR
int write_3d_uint8(char *keyword, ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t yN, ptrdiff_t zN, uint8_t *block);

// Write a binned 3D array in one file, datatype double / H5T_IEEE_F64LE
int write_3d_double_in_one(char *keyword, ptrdiff_t xN, ptrdiff_t yN, ptrdiff_t zN, double *array);

// Write 1D data (FSC or PRTF) in plain text
int write_profile(char *filename, ptrdiff_t kN, double *profile);

// str2: the asterisk in str1 is replaced by number; used in merge.c and prtf.c
void replace(char *str1, char *str2, ptrdiff_t number);

// Remove leading and trailing spaces and tabs from a string.; used in read_config()
void trim(char string[]);

// Read configurations
int read_config(char *path_config,
    ptrdiff_t *xN, ptrdiff_t *yN, ptrdiff_t *zN,
    int *need_R0, int *need_S, int *need_H,
    char *keyword_M, char *keyword_R0, char *keyword_S, char *keyword_H, char *keyword_Rp,
    char *method, double *beta, double *lower_bound, double *upper_bound,
    int *num_loop, int *num_iteration,
    double *sigma0, double *sigmar, double *th);
//

#endif
