#pragma once

#include "coefficients.h"

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/cuda/execution_policy.h>


/**
 * \brief Deriche causal or anticausal pass
 * \param stream            Cuda Stream to execute this task on
 * \param c                 coefficients precomputed by deriche_precomp()
 * \param buffer_begin      workspace array with space for at least height * width elements
 * \param src_begin         data to be convolved
 * \param src_end           end of data to be convolved
 * \param height            dimension 0 of input data
 * \param width             dimension 1 of input data
 * \param row_stride        stride along dimension 0 (i.e. &src(i + 1, j) - &src(i, j) = row_stride)
 * \param column_stride     stride along dimension 1 (i.e. &src(i, j + 1) - &src(i, j) = column_stride)
 */
template <class T,
  class TmpIt, class InIt, bool anticausal>
void deriche_thrust_pass(
    const cudaStream_t& stream,
    const deriche_coeffs<T>& c,
    TmpIt buffer_begin,
    InIt src_begin,
    unsigned int height, unsigned int width,
    unsigned int row_stride, unsigned int column_stride)
{
  switch(anticausal)
  {
    case false:
      // causal pass
      thrust::for_each_n(thrust::cuda::par.on(stream),
                         thrust::counting_iterator<int>(0), height,
                         [buffer_begin, src_begin, row_stride, column_stride, width, c] __device__ (int n) {
                         auto row = buffer_begin + n * row_stride;
                         auto src = src_begin + n * row_stride;

                         // init recursion
                         for(int i = 0; i < 4; ++i)
                         {
                         row[i * column_stride] = 0;
                         for(int j = 0; j <= i; ++j)
                         {
                         row[i * column_stride] = row[i * column_stride] + c.b_causal[j] * src[(i - j) * column_stride];
                         }
                         for(int j = 1; j <= i; ++j)
                         {
                         row[i * column_stride] = row[i * column_stride] - c.a[j] * row[(i - j) * column_stride];
                         }
                         }

                         // recurse
                         for(int i = 4; i < width; ++i)
                         {
                           row[i * column_stride] = 
                             c.b_causal[0] * src[(i - 0) * column_stride]
                             + c.b_causal[1] * src[(i - 1) * column_stride]
                             + c.b_causal[2] * src[(i - 2) * column_stride]
                             + c.b_causal[3] * src[(i - 3) * column_stride]
                             - c.a[1] * row[(i - 1) * column_stride]
                             - c.a[2] * row[(i - 2) * column_stride]
                             - c.a[3] * row[(i - 3) * column_stride]
                             - c.a[4] * row[(i - 4) * column_stride]
                             ; 
                         }
                         });
      break;
    case true:
      // anticausal pass
      thrust::for_each_n(thrust::cuda::par.on(stream),
                         thrust::counting_iterator<int>(0), height,
                         [buffer_begin, src_begin, row_stride, column_stride, width, c] __device__ (int n) {
                         auto row = buffer_begin + n * row_stride;
                         auto src = src_begin + n * row_stride;

                         // init recursion
                         for(int i = 0; i < 4; ++i)
                         {
                         row[i * column_stride] = 0;
                         for(int j = 1; j <= i; ++j)
                         {
                         row[i * column_stride] = row[i * column_stride] + c.b_anticausal[j] * src[((width - 1) - i + j) * column_stride];
                         }
                         for(int j = 1; j <= i; ++j)
                         {
                         row[i * column_stride] = row[i * column_stride] - c.a[j] * row[(i - j) * column_stride];
                         }
                         }

                         // recurse
                         for(int i = 4; i < width; ++i)
                         {
                           row[i * column_stride] = 
                             c.b_anticausal[1] * src[((width - 1) - i + 1) * column_stride]
                             + c.b_anticausal[2] * src[((width - 1) - i + 2) * column_stride]
                             + c.b_anticausal[3] * src[((width - 1) - i + 3) * column_stride]
                             + c.b_anticausal[4] * src[((width - 1) - i + 4) * column_stride]
                             - c.a[1] * row[(i - 1) * column_stride]
                             - c.a[2] * row[(i - 2) * column_stride]
                             - c.a[3] * row[(i - 3) * column_stride]
                             - c.a[4] * row[(i - 4) * column_stride]
                             ;
                         }
                         });
      break;
  }
}

/**
 * \brief Deriche Gaussian convolution 2D parallelized over rows
 * \param c                 coefficients precomputed by deriche_precomp()
 * \param dest_begin        output convolved data
 * \param buffer_l_begin    workspace array with space for at least height * width elements
 * \param buffer_r_begin    workspace array with space for at least height * width elements
 * \param src_begin         data to be convolved
 * \param src_end           end of data to be convolved
 * \param height            dimension 0 of input data
 * \param width             dimension 1 of input data
 * \param row_stride        stride along dimension 0 (i.e. &src(i + 1, j) - &src(i, j) = row_stride)
 * \param column_stride     stride along dimension 1 (i.e. &src(i, j + 1) - &src(i, j) = column_stride)
 */
template <class T,
  class OutIt, class TmpIt, class InIt>
void deriche_thrust_2d(
    const deriche_coeffs<T>& c,
    OutIt dest_begin,
    TmpIt buffer_l_begin,
    TmpIt buffer_r_begin,
    InIt src_begin,
    unsigned int height, unsigned int width,
    unsigned int row_stride, unsigned int column_stride)
{
    assert(c.K == 4);

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // causal
    deriche_thrust_pass<T, TmpIt, InIt, false>(
        s1, c, buffer_l_begin, src_begin, height, width, row_stride, column_stride);
    // anticausal
    deriche_thrust_pass<T, TmpIt, InIt, true>(
        s2, c, buffer_r_begin, src_begin, height, width, row_stride, column_stride);

    /*
    thrust::transform(
        buffer_l_begin, buffer_l_begin + width * height,
        thrust::make_reverse_iterator(buffer_r_begin + width * height),
        dest_begin,
        thrust::plus<T>());
        */
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    thrust::for_each_n(thrust::counting_iterator<int>(0), height,
                       [dest_begin, buffer_l_begin, buffer_r_begin, row_stride, column_stride, width] __device__ (int n) {
                       auto dest = dest_begin + n * row_stride;
                       auto row_l = buffer_l_begin + n * row_stride;
                       auto row_r = buffer_r_begin + n * row_stride;
                       for(int i = 0; i < width; ++i)
                       {
                         dest[i * column_stride] = row_l[i * column_stride] + row_r[(width - 1 - i) * column_stride];
                       }
                       });
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    return;
}
