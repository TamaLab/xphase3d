#ifndef RECONS_H
#define RECONS_H
#include <stddef.h>
#include <stdint.h>
#include "fftw3-mpi.h"


// Forward Fourier Transform, from RC to F
int fft(fftw_plan plan_fft);
// Backward Fourier Transform, from F to RC
int ifft(fftw_plan plan_ifft, ptrdiff_t indexN, ptrdiff_t local_indexN, fftw_complex *local_RC);

// Projection to satisfy M constraint
int projM(fftw_plan plan_fft, fftw_plan plan_ifft,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex* local_RC, fftw_complex *local_F, double *local_M, uint8_t *local_H);
// Projection to satisfy S constraint
int projS(double lower_bound, double upper_bound, ptrdiff_t local_indexN, fftw_complex *local_RC, uint8_t *local_S);
// Reflection by S constraint
int reflS(double lower_bound, double upper_bound, ptrdiff_t local_indexN, fftw_complex *local_RC, uint8_t *local_S);

// Backup the real part of RC to RC_in; used within optimizations
int copy_local_RC_real(ptrdiff_t local_indexN, fftw_complex *local_RC, double *local_RC_in);

// Hybrid Input-Output (HIO)
int optHIO(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H);

// Error Reduction (ER)
// beta is not needed; to be consistent with other optimization methods
int optER(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H);

// Averaged Successive Reflections (ASR)
// beta is not needed; to be consistent with other optimization methods
int optASR(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H);

// Hybrid Projection Reflection (HPR)
int optHPR(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H);

// Relaxed Averaged Alternating Reflectors (RAAR)
int optRAAR(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H);

// Difference Map (DM)
int optDM(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H);

// Shrink wrap: Gaussian kernel
int get_kernel(ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t xN,
    ptrdiff_t yN, ptrdiff_t zN, double sigma, fftw_complex *local_kernel);

// Shrink wrap: Gaussian convolution of the model |RC|
int get_R_conv(ptrdiff_t indexN, ptrdiff_t local_indexN, fftw_complex *local_RC,
    fftw_plan plan_fft_kernel, fftw_plan plan_fft_R_abs, fftw_plan plan_ifft_R_conv,
    fftw_complex *local_R_abs, fftw_complex *local_R_conv,
    fftw_complex *local_F_kernel, fftw_complex *local_F_R_abs, fftw_complex *local_F_R_conv);

// Shrink wrap: update the support
int update_S(double th, ptrdiff_t local_indexN, uint8_t *local_S, fftw_complex *local_R_conv, MPI_Comm communicator);

#endif
