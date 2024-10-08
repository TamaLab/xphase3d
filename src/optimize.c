#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "optimize.h"

// Forward Fourier Transform, from RC to F
int fft(fftw_plan plan_fft, double *t_iter_fft)
{
    clock_t c_start, c_end;
    c_start = clock();
    fftw_execute(plan_fft);
    c_end = clock();
    *t_iter_fft += (double)(c_end - c_start);
    return 0;
}

// Backward Fourier Transform, from F to RC
int ifft(fftw_plan plan_ifft, ptrdiff_t indexN, ptrdiff_t local_indexN, fftw_complex *local_RC, double *t_iter_for, double *t_iter_fft)
{
    clock_t c_start, c_end;
    c_start = clock();
    fftw_execute(plan_ifft);
    c_end = clock();
    *t_iter_fft += (double)(c_end - c_start);
    // Normalization
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC[i][0] = local_RC[i][0] / indexN;
        local_RC[i][1] = local_RC[i][1] / indexN;
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);
    return 0;
}

// Projection to satisfy M constraint
int projM(fftw_plan plan_fft, fftw_plan plan_ifft,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex* local_RC, fftw_complex *local_F, double *local_M, uint8_t *local_H,
    double *t_iter_for, double *t_iter_fft)
{
    // Note: The imaginary part of RC should be cast away prior to fft
    fft(plan_fft, t_iter_fft);

    // Update F outside the missing region H
    double F_modulus, cos_value, sin_value;

    clock_t c_start, c_end;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        if (local_H[i] == 0)
        {
            // When complex number is zero, the angle is defined as 0
            F_modulus = sqrt(local_F[i][0]*local_F[i][0] + local_F[i][1]*local_F[i][1]);
            if (F_modulus < 1e-10)
            {
                cos_value = 1.0;
                sin_value = 0.0;
            }
            else
            {
                cos_value = local_F[i][0] / F_modulus;
                sin_value = local_F[i][1] / F_modulus;
            }
            local_F[i][0] = local_M[i] * cos_value;
            local_F[i][1] = local_M[i] * sin_value;
        }
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    // Update model in real space
    ifft(plan_ifft, indexN, local_indexN, local_RC, t_iter_for, t_iter_fft);

    return 0;
}

// Projection to satisfy S constraint
int projS(double lower_bound, double upper_bound, ptrdiff_t local_indexN, fftw_complex *local_RC, uint8_t *local_S, double *t_iter_for)
{
    clock_t c_start, c_end;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        // Outside support, to 0
        if (local_S[i] == 0)
        {
            local_RC[i][0] = 0.0;
        }

        // Inside support
        else
        {
            // Density upper bound
            if (local_RC[i][0] > upper_bound)
            {
                local_RC[i][0] = upper_bound;
            }
            // Density lower bound
            if (local_RC[i][0] < lower_bound)
            {
                local_RC[i][0] = lower_bound;
            }
        }

        // Cast away imaginary part of RC
        local_RC[i][1] = 0.0;
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    return 0;
}

// Reflection by S constraint
int reflS(double lower_bound, double upper_bound, ptrdiff_t local_indexN, fftw_complex *local_RC, uint8_t *local_S, double *t_iter_for)
{
    clock_t c_start, c_end;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        // Outside support
        if (local_S[i] == 0)
        {
            local_RC[i][0] = - local_RC[i][0];
        }
        // Inside support
        else
        {
            // Density upper bound
            if (local_RC[i][0] > upper_bound)
            {
                local_RC[i][0] = 2.0 * upper_bound - local_RC[i][0];
            }
            // Density lower bound
            if (local_RC[i][0] < lower_bound)
            {
                local_RC[i][0] = 2.0 * lower_bound - local_RC[i][0];
            }
        }

        // Cast away imaginary part of RC
        local_RC[i][1] = 0.0;
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    return 0;
}

// Backup the real part of RC to RC_in; used in optimization functions
int copy_local_RC_real(ptrdiff_t local_indexN, fftw_complex *local_RC, double *local_RC_in, double *t_iter_for)
{
    clock_t c_start, c_end;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC_in[i] = local_RC[i][0];
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    return 0;
}

// Hybrid Input-Output (HIO)
int optHIO(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H,
    double *t_iter_for, double *t_iter_fft)
{
    // Store RC input
    double *local_RC_in = (double*)malloc(sizeof(double) * local_indexN);
    copy_local_RC_real(local_indexN, local_RC, local_RC_in, t_iter_for);

    // M projection, local_RC is output
    projM(plan_fft, plan_ifft, indexN, local_indexN, local_RC, local_F, local_M, local_H, t_iter_for, t_iter_fft);

    // HIO optimization
    clock_t c_start, c_end;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        // Outside support
        if (local_S[i] == 0)
        {
            local_RC[i][0] = local_RC_in[i] - beta * local_RC[i][0];
        }
        // Inside support
        else
        {
            // Density upper bound
            if (local_RC[i][0] > upper_bound)
            {
                local_RC[i][0] = local_RC_in[i] - beta * (local_RC[i][0] - upper_bound);
            }
            // Density lower bound
            if (local_RC[i][0] < lower_bound)
            {
                local_RC[i][0] = local_RC_in[i] - beta * (local_RC[i][0] - lower_bound);
            }
        }

        // Cast away imaginary part of RC
        local_RC[i][1] = 0.0;
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    free(local_RC_in);
    return 0;
}

// Error Reduction (ER)
// beta is not needed; to be consistent with other optimization methods
int optER(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H,
    double *t_iter_for, double *t_iter_fft)
{
    // M projection, local_RC is output
    projM(plan_fft, plan_ifft, indexN, local_indexN, local_RC, local_F, local_M, local_H, t_iter_for, t_iter_fft);
    // S projection, local_RC is output
    // Cast away imaginary part of RC
    projS(lower_bound, upper_bound, local_indexN, local_RC, local_S, t_iter_for);
    beta += 0;
    return 0;
}

// Averaged Successive Reflections (ASR)
// beta is not needed; to be consistent with other optimization methods
int optASR(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H,
    double *t_iter_for, double *t_iter_fft)
{
    // Store RC input
    double *local_RC_in = (double*)malloc(sizeof(double) * local_indexN);
    copy_local_RC_real(local_indexN, local_RC, local_RC_in, t_iter_for);

    // M projection
    projM(plan_fft, plan_ifft, indexN, local_indexN, local_RC, local_F, local_M, local_H, t_iter_for, t_iter_fft);

    // M reflection
    clock_t c_start, c_end;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC[i][0] = 2.0 * local_RC[i][0] - local_RC_in[i];
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    // S reflection
    // Cast away imaginary part of RC
    reflS(lower_bound, upper_bound, local_indexN, local_RC, local_S, t_iter_for);

    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC[i][0] = 0.5 * (local_RC[i][0] + local_RC_in[i]);
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    free(local_RC_in);
    beta += 0;
    return 0;
}

// Hybrid Projection Reflection (HPR)
int optHPR(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H,
    double *t_iter_for, double *t_iter_fft)
{
    // Store RC input
    double *local_RC_in = (double*)malloc(sizeof(double) * local_indexN);
    copy_local_RC_real(local_indexN, local_RC, local_RC_in, t_iter_for);

    // M projection
    projM(plan_fft, plan_ifft, indexN, local_indexN, local_RC, local_F, local_M, local_H, t_iter_for, t_iter_fft);

    // Store M projection
    double *local_RC_pM = (double*)malloc(sizeof(double) * local_indexN);
    copy_local_RC_real(local_indexN, local_RC, local_RC_pM, t_iter_for);

    // beta
    clock_t c_start, c_end;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        // M reflection
        local_RC[i][0] = 2.0 * local_RC[i][0] - local_RC_in[i];
        // reflection_M + (beta - 1.0) * projection_M
        local_RC[i][0] = local_RC[i][0] + (beta - 1.0) * local_RC_pM[i];
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    // S reflection
    // Cast away imaginary part of RC
    reflS(lower_bound, upper_bound, local_indexN, local_RC, local_S, t_iter_for);

    // Optimized RC
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC[i][0] = 0.5 * (local_RC[i][0] + local_RC_in[i] + (1.0 - beta) * local_RC_pM[i]);
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    free(local_RC_in);
    free(local_RC_pM);
    return 0;
}

// Relaxed Averaged Alternating Reflectors (RAAR)
int optRAAR(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H,
    double *t_iter_for, double *t_iter_fft)
{
    // Store RC input
    double *local_RC_in = (double*)malloc(sizeof(double) * local_indexN);
    copy_local_RC_real(local_indexN, local_RC, local_RC_in, t_iter_for);

    // M projection
    projM(plan_fft, plan_ifft, indexN, local_indexN, local_RC, local_F, local_M, local_H, t_iter_for, t_iter_fft);

    // Store M projection
    double *local_RC_pM = (double*)malloc(sizeof(double) * local_indexN);
    copy_local_RC_real(local_indexN, local_RC, local_RC_pM, t_iter_for);

    // M reflection
    clock_t c_start, c_end;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC[i][0] = 2.0 * local_RC[i][0] - local_RC_in[i];
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    // S reflection
    // Cast away imaginary part of RC
    reflS(lower_bound, upper_bound, local_indexN, local_RC, local_S, t_iter_for);

    // Optimized RC
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC[i][0] = 0.5 * beta * (local_RC[i][0] + local_RC_in[i]) + (1.0 - beta) * local_RC_pM[i];
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    free(local_RC_in);
    free(local_RC_pM);
    return 0;
}

// Difference Map (DM)
int optDM(fftw_plan plan_fft, fftw_plan plan_ifft,
    double beta, double lower_bound, double upper_bound,
    ptrdiff_t indexN, ptrdiff_t local_indexN,
    fftw_complex *local_RC, fftw_complex *local_F,
    double *local_M, uint8_t *local_S, uint8_t *local_H,
    double *t_iter_for, double *t_iter_fft)
{
    double s, m;
    s = 1.0 / beta;
    m = - s;

    // Store RC input
    double *local_RC_in = (double*)malloc(sizeof(double) * local_indexN);
    copy_local_RC_real(local_indexN, local_RC, local_RC_in, t_iter_for);

    // M projection first, then linear combination with s, and finally S projection: spS
    projM(plan_fft, plan_ifft, indexN, local_indexN, local_RC, local_F, local_M, local_H, t_iter_for, t_iter_fft);
    clock_t c_start, c_end;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC[i][0] = (1.0 + s) * local_RC[i][0] - s * local_RC_in[i];
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);
    projS(lower_bound, upper_bound, local_indexN, local_RC, local_S, t_iter_for);
    // Store spS
    double *local_RC_spS = (double*)malloc(sizeof(double) * local_indexN);
    copy_local_RC_real(local_indexN, local_RC, local_RC_spS, t_iter_for);

    // Recover local_RC to input
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC[i][0] = local_RC_in[i];
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    // S projection first, then linear combination with m, and finally M projection: mpM
    projS(lower_bound, upper_bound, local_indexN, local_RC, local_S, t_iter_for);
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC[i][0] = (1.0 + m) * local_RC[i][0] - m * local_RC_in[i];
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);
    projM(plan_fft, plan_ifft, indexN, local_indexN, local_RC, local_F, local_M, local_H, t_iter_for, t_iter_fft);

    // Optimized RC: RC_in + beta * RC_sPs + beta * RC_mPm (RC)
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_RC[i][0] = local_RC_in[i] + beta * local_RC_spS[i] - beta * local_RC[i][0];
        // Cast away imaginary part of RC
        local_RC[i][1] = 0.0;
    }
    c_end = clock();
    *t_iter_for += (double)(c_end - c_start);

    free(local_RC_in);
    free(local_RC_spS);
    return 0;
}

// Shrink wrap: Gaussian kernel
// The center is at [0,0,0]
int get_kernel(ptrdiff_t x_start, ptrdiff_t x_end, ptrdiff_t xN, ptrdiff_t yN, ptrdiff_t zN, double sigma, fftw_complex *local_kernel, double *t_wrap_for)
{
    int index;
    // mu has only integer part, no decimal part
    double mu_x, mu_y, mu_z;
    mu_x = xN / 2;
    mu_y = yN / 2;
    mu_z = zN / 2;
    // Shift to put mu at the first element
    double shift_x, shift_y, shift_z;
    shift_x = xN - mu_x;
    shift_y = yN - mu_y;
    shift_z = zN - mu_z;
    double value_x, value_y, value_z;
    // Amplitude
    double amp, power;
    amp = 1 / (pow(2*M_PI, 1.5) * sigma*sigma*sigma);

    clock_t c_start, c_end;
    c_start = clock();
    for (int x = x_start; x < x_end; x++)
    {
        value_x = x - shift_x;
        if (value_x < 0.0)
        {
            value_x = value_x + xN;
        }

        for (int y = 0; y < yN; y++)
        {
            value_y = y - shift_y;
            if (value_y < 0.0)
            {
                value_y = value_y + yN;
            }

            for (int z = 0; z < zN; z++)
            {
                value_z = z - shift_z;
                if (value_z < 0.0)
                {
                    value_z = value_z + zN;
                }

                power = - ( (value_x - mu_x) * (value_x - mu_x)
                    + (value_y - mu_y) * (value_y - mu_y)
                    + (value_z - mu_z) * (value_z - mu_z) )
                    / (2 * sigma*sigma);
                index = (x - x_start)*yN*zN + y*zN + z;
                local_kernel[index][0] = amp * exp(power);
                local_kernel[index][1] = 0;
            }
        }
    }
    c_end = clock();
    *t_wrap_for += (double)(c_end - c_start);

    return 0;
}

// Shrink wrap: Gaussian convolution of the model |RC|
int get_R_conv(ptrdiff_t indexN, ptrdiff_t local_indexN, fftw_complex *local_RC,
    fftw_plan plan_fft_kernel, fftw_plan plan_fft_R_abs, fftw_plan plan_ifft_R_conv,
    fftw_complex *local_R_abs, fftw_complex *local_R_conv,
    fftw_complex *local_F_kernel, fftw_complex *local_F_R_abs, fftw_complex *local_F_R_conv,
    double *t_wrap_for, double *t_wrap_fft)
{
    // R_abs = |RC|
    clock_t c_start, c_end;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_R_abs[i][0] = sqrt(local_RC[i][0] * local_RC[i][0] + local_RC[i][1] * local_RC[i][1]);
        local_R_abs[i][1] = 0.0;
    }
    c_end = clock();
    *t_wrap_for += (double)(c_end - c_start);

    c_start = clock();
    fftw_execute(plan_fft_kernel);
    fftw_execute(plan_fft_R_abs);
    c_end = clock();
    *t_wrap_fft += (double)(c_end - c_start);

    // Multiplication in Fourier space
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_F_R_conv[i][0] = local_F_kernel[i][0] * local_F_R_abs[i][0] - local_F_kernel[i][1] * local_F_R_abs[i][1];
        local_F_R_conv[i][1] = local_F_kernel[i][0] * local_F_R_abs[i][1] + local_F_kernel[i][1] * local_F_R_abs[i][0];
    }
    c_end = clock();
    *t_wrap_for += (double)(c_end - c_start);

    // ifft and normalization
    c_start = clock();
    fftw_execute(plan_ifft_R_conv);
    c_end = clock();
    *t_wrap_fft += (double)(c_end - c_start);

    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        local_R_conv[i][0] = local_R_conv[i][0] / indexN;
        local_R_conv[i][1] = 0.0;
    }
    c_end = clock();
    *t_wrap_for += (double)(c_end - c_start);

    return 0;
}

// Shrink wrap: update the support
int update_S(double th, ptrdiff_t local_indexN, uint8_t *local_S, fftw_complex *local_R_conv, MPI_Comm communicator, double *t_wrap_for)
{
    double local_max, global_max, threshold;
    local_max = 0.0;

    clock_t c_start, c_end;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        if (local_R_conv[i][0] > local_max)
        {
            local_max = local_R_conv[i][0];
        }
    }
    c_end = clock();
    *t_wrap_for += (double)(c_end - c_start);

    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, communicator);

    threshold = global_max * th;
    c_start = clock();
    for (ptrdiff_t i = 0; i < local_indexN; i++)
    {
        if (local_R_conv[i][0] > threshold)
        {
            local_S[i] = 1;
        }
        else
        {
            local_S[i] = 0;
        }
    }
    c_end = clock();
    *t_wrap_for += (double)(c_end - c_start);

    return 0;
}
