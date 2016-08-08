/*
  convolve_iir_gpu function for convolving an image horizontally using the thrust library
*/
#include "fastfilters.hxx"

#include <string>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>

#include <thrust/functional.h>
#include <thrust/tabulate.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

typedef thrust::tuple<unsigned int, float, float, float, float> InputTuple;
typedef thrust::tuple<float, float, float, float> FloatTuple;

struct convolve_4
{
  const FloatTuple d;

  convolve_4( FloatTuple _d ) : d(_d) {}

  template <typename Tuple, typename Tuple2>
  __host__ __device__
  Tuple operator()(const Tuple &y, const Tuple2 &x)
  {
    using thrust::get;

    unsigned int A_exponent = get<0>(x);

    float A[4][4];
    for( int i=0; i<4; ++i )
      {
        for( int j=0; j<4; ++j )
          {
            A[i][j] = ((i+1) == j)? 1 : 0; // realizes transition
          }
      }
    A[0][0] = - get<0>(d);
    A[1][0] = - get<1>(d);
    A[2][0] = - get<2>(d);
    A[3][0] = - get<3>(d);

    // naive calculation of matrix powers
    for( int p=1; p<A_exponent; ++p )
      {
        for( int i=0; i<4; ++i)
          {
	   float sum = - get<0>(d) * A[i][0]
                       - get<1>(d) * A[i][1]
                       - get<2>(d) * A[i][2]
                       - get<3>(d) * A[i][3];
            A[i][3] = A[i][2];
            A[i][2] = A[i][1];
            A[i][1] = A[i][0];
            A[i][0] = sum;
          }
      }

    // calculate y*A
    float yA[4]; // y*A
    for( int j=0; j<4; ++j)
      {
        yA[j] = get<1>(y) * A[0][j]
              + get<2>(y) * A[1][j]
              + get<3>(y) * A[2][j]
              + get<4>(y) * A[3][j];
      }
  
    return thrust::make_tuple( get<0>(y) + A_exponent,
                               yA[0]+get<1>(x), 
                               yA[1]+get<2>(x),
                               yA[2]+get<3>(x), 
                               yA[3]+get<4>(x) );
  }
};


struct addseconds : public thrust::binary_function<InputTuple, InputTuple, float>
{
  __host__ __device__
  float operator()( const InputTuple& x, const InputTuple& y ) const
  {
    return thrust::get<1>(x) + thrust::get<1>(y);
  }
};


struct row_idx : public thrust::unary_function<int, int>
{
  const int width;

  row_idx(int _width) : width(_width)  {}

  __host__ __device__
  int operator()( const int idx ) const
  {
    return (idx) / width;
  }
};

struct make_b: public thrust::unary_function<FloatTuple, float>
{
  const FloatTuple n;

  make_b(FloatTuple _n) : n(_n)  {}

  __host__ __device__
  float operator()( const FloatTuple& x ) const
  {
    return thrust::get<0>(n) * thrust::get<0>(x)
         + thrust::get<1>(n) * thrust::get<1>(x)
         + thrust::get<2>(n) * thrust::get<2>(x)
         + thrust::get<3>(n) * thrust::get<3>(x);
  }
};

void convolve_iir_gpu(const float* input, float* output, const unsigned int n_cols, const unsigned int n_rows,
                      const fastfilters::iir::Coefficients &coefs )
{
  cudaSetDevice(0);
  unsigned int n_pixels, n_border;

  n_pixels = n_cols * n_rows;
  n_border = 3; // since window size = 4

  FloatTuple d(coefs.d[0], coefs.d[1],
               coefs.d[2], coefs.d[3]);
  FloatTuple causal(coefs.n_causal[0], coefs.n_causal[1],
                    coefs.n_causal[2], coefs.n_causal[3]);
  FloatTuple anticausal(coefs.n_anticausal[0], coefs.n_anticausal[1],
                        coefs.n_anticausal[2], coefs.n_anticausal[3]);
 
     
  thrust::device_vector<float> in(n_pixels+2*n_border); // left and right border
  thrust::device_vector<float> out(n_pixels);

  // pad border with zeros
  thrust::fill(in.begin(), in.begin()+n_border, 0.0);
  thrust::copy_n(input, n_pixels, in.begin()+n_border);
  thrust::fill(in.begin()+n_border+n_pixels, in.end(), 0.0);

  thrust::device_vector<InputTuple> forth(n_pixels);
  thrust::device_vector<InputTuple> back(n_pixels);

  thrust::device_vector<int> key(n_pixels);
  thrust::tabulate(key.begin(), key.end(), row_idx(n_cols));
  thrust::equal_to<int> eq;
  
  auto itertuple = thrust::make_tuple(in.begin()+3, in.begin()+2,
                                      in.begin()+1, in.begin());
  auto iterb = thrust::make_transform_iterator(thrust::make_zip_iterator(itertuple),
                                               make_b(causal));
  auto iter = thrust::make_zip_iterator
    (thrust::make_tuple (thrust::make_constant_iterator<int>(1),
                         iterb,
                         thrust::make_constant_iterator<float>(0.0),
                         thrust::make_constant_iterator<float>(0.0),
                         thrust::make_constant_iterator<float>(0.0)));                                 
  inclusive_scan_by_key(key.begin(),
                        key.end(),
                        iter,
                        forth.begin(),
                        eq,
                        convolve_4(d));
  
  
  auto ritertuple = thrust::make_tuple(in.rbegin()+3, in.rbegin()+2,
                                      in.rbegin()+1, in.rbegin());
  auto riterb = thrust::make_transform_iterator(thrust::make_zip_iterator(ritertuple),
                                               make_b(anticausal));
  auto riter = thrust::make_zip_iterator
    (thrust::make_tuple (thrust::make_constant_iterator<int>(1),
                         riterb,
                         thrust::make_constant_iterator<float>(0.0),
                         thrust::make_constant_iterator<float>(0.0),
                         thrust::make_constant_iterator<float>(0.0)));                                 

  thrust::inclusive_scan_by_key
    (key.begin(),
     key.end(),
     riter,
     back.begin(),
     eq,
     convolve_4(d));

  thrust::transform(forth.begin(), forth.end(), back.rbegin(), out.begin(), addseconds());
  
  thrust::copy(out.begin(), out.end(), output);
}

