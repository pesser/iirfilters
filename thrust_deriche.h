#pragma once

/**
 * Thrust implementations of deriche convolution. *
 **/

#include "coefficients.h"

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cublas_v2.h>


#ifndef cublasSafeCall
#define cublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)
#endif


void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
	if(CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__,err); 
		cudaDeviceReset();
    assert(0); 
	}
}


struct transpose_index : public thrust::unary_function<int, int>
{
  int m, n;

  __host__ __device__
    transpose_index(int m, int n) : m(m), n(n) {}

  __host__ __device__
    int operator()(int linear_index)
    {
      int i = linear_index / n;
      int j = linear_index % n;

      return m * j + i;
    }
};


/**
 * \brief Deriche causal or anticausal pass depending on last template parameter.
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
  assert(c.K == 4);
  switch(anticausal)
  {
    case false:
      // causal pass
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
      thrust::for_each_n(thrust::cuda::par.on(stream),
#else
      thrust::for_each_n(
#endif
                         thrust::counting_iterator<int>(0), height,
                         [buffer_begin, src_begin, row_stride, column_stride, width, c]
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
                         __device__
#endif
                         (int n) {
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
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
      thrust::for_each_n(thrust::cuda::par.on(stream),
#else
      thrust::for_each_n(
#endif
                         thrust::counting_iterator<int>(0), height,
                         [buffer_begin, src_begin, row_stride, column_stride, width, c]
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
                         __device__
#endif
                         (int n) {
                         auto row = buffer_begin + n * row_stride;
                         auto src = src_begin + n * row_stride;

                         // init recursion
                         for(int i = 0; i < 4; ++i)
                         {
                         row[(width - 1 - i) * column_stride] = 0;
                         for(int j = 1; j <= i; ++j)
                         {
                         row[(width - 1 - i) * column_stride] = row[(width - 1 - i) * column_stride]
                                                              + c.b_anticausal[j] * src[((width - 1) - i + j) * column_stride];
                         }
                         for(int j = 1; j <= i; ++j)
                         {
                         row[(width - 1 - i) * column_stride] = row[(width - 1 - i) * column_stride] - c.a[j] * row[(width - 1 - (i - j)) * column_stride];
                         }
                         }

                         // recurse
                         for(int i = 4; i < width; ++i)
                         {
                           row[(width - 1 - i) * column_stride] = 
                               c.b_anticausal[1] * src[((width - 1) - i + 1) * column_stride]
                             + c.b_anticausal[2] * src[((width - 1) - i + 2) * column_stride]
                             + c.b_anticausal[3] * src[((width - 1) - i + 3) * column_stride]
                             + c.b_anticausal[4] * src[((width - 1) - i + 4) * column_stride]
                             - c.a[1] * row[(width - 1 - (i - 1)) * column_stride]
                             - c.a[2] * row[(width - 1 - (i - 2)) * column_stride]
                             - c.a[3] * row[(width - 1 - (i - 3)) * column_stride]
                             - c.a[4] * row[(width - 1 - (i - 4)) * column_stride]
                             ;
                         }
                         });
      break;
  }
}

/**
 * \brief Deriche Gaussian convolution single 2D pass. Result is transposed.
 * \param c                 coefficients precomputed by deriche_precomp()
 * \param dest_begin        output convolved data
 * \param buffer_l_begin    workspace array with space for at least height * width elements
 * \param buffer_r_begin    workspace array with space for at least height * width elements
 * \param src_begin         data to be convolved
 * \param height            dimension 0 of input data
 * \param width             dimension 1 of input data
 * \param row_stride        stride along dimension 0 (i.e. &src(i + 1, j) - &src(i, j) = row_stride)
 * \param column_stride     stride along dimension 1 (i.e. &src(i, j + 1) - &src(i, j) = column_stride)
 */
template <class T,
  class OutIt, class TmpIt, class InIt>
void deriche_thrust_2d_transpose_pass(
    const deriche_coeffs<T>& c,
    OutIt dest_begin,
    TmpIt buffer_l_begin,
    TmpIt buffer_r_begin,
    InIt src_begin,
    unsigned int height, unsigned int width,
    unsigned int row_stride, unsigned int column_stride,
    const cudaStream_t& s1,
    const cudaStream_t& s2,
    cublasHandle_t* handle = nullptr
    )
{
    // causal
    deriche_thrust_pass<T, TmpIt, InIt, false>(
        s1, c, buffer_l_begin, src_begin, height, width, row_stride, column_stride);
    // anticausal
    deriche_thrust_pass<T, TmpIt, InIt, true>(
        s2, c, buffer_r_begin, src_begin, height, width, row_stride, column_stride);

    // sync
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
#endif

    // sum up transpose of causal and anticausal pass
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    assert(handle != nullptr);
    T* buffer_l_ptr  = thrust::raw_pointer_cast(&*buffer_l_begin);
    T* buffer_r_ptr  = thrust::raw_pointer_cast(&*buffer_r_begin);
    T* dest_ptr = thrust::raw_pointer_cast(&*dest_begin);
    T alpha = 1.;
    T beta  = 1.;
    cublasSafeCall(cublasSgeam(*handle, CUBLAS_OP_T, CUBLAS_OP_T, height, width, &alpha, buffer_l_ptr, width, &beta, buffer_r_ptr, width, dest_ptr, height));
#else
    auto transposed_dest_begin =
      thrust::make_permutation_iterator(dest_begin, thrust::make_transform_iterator(thrust::make_counting_iterator(0), transpose_index(height, width)));
    thrust::transform(
        buffer_l_begin, buffer_l_begin + width * height,
        buffer_r_begin,
        transposed_dest_begin,
        thrust::plus<T>());
#endif

    return;
}


/**
 * \brief Deriche Gaussian convolution 2D. Only for order 4 and data larger
 * than 4 elements in each dimension. Vertical then horizontal.
 * \param c                 coefficients precomputed by deriche_precomp()
 * \param dest_begin        output convolved data
 * \param buffer_l_begin    workspace array with space for at least height * width elements
 * \param buffer_r_begin    workspace array with space for at least height * width elements
 * \param src_begin         data to be convolved
 * \param height            dimension 0 of input data
 * \param width             dimension 1 of input data
 * \param s1                cuda stream
 * \param s2                cuda stream
 * \param handle            cublas handle
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
    const cudaStream_t& s1,
    const cudaStream_t& s2,
    cublasHandle_t* handle = nullptr
    )
{
  // vertical pass & transpose
  deriche_thrust_2d_transpose_pass<T>(
      c,
      dest_begin,
      buffer_l_begin, buffer_r_begin,
      src_begin, width, height, 1, width,
      s1, s2, handle
      );
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
  cudaDeviceSynchronize();
#endif
  // vertical pass & transpose
  deriche_thrust_2d_transpose_pass<T>(
      c,
      dest_begin,
      buffer_l_begin, buffer_r_begin,
      dest_begin, height, width, 1, height,
      s1, s2, handle
      );
}
