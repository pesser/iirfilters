/*
  Program to time functions convolving images using the thrust library 
  Prints time in nanoseconds

  Usage: ./iirfilter-thrust pngfile
*/
#include "fastfilters.hxx"
#include "image.hxx"

#include <string>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/tuple.h>

typedef thrust::tuple<float, float, float, float> FloatTuple;

template <typename Tuple>
struct addfirsts : public thrust::binary_function<Tuple,Tuple,float>
{
  __host__ __device__
  Tuple operator()( const Tuple& x, const Tuple& y ) const
  {
    return thrust::get<0>(x) + thrust::get<0>(y);
  }
};

struct row_idx : public thrust::unary_function<int, int>
{
  const int width;
  
  row_idx(Tuple _width) : width(_width)  {}
  
  __host__ __device__
  Tuple operator()( const int idx ) const
  {
    return (idx) % width;
  }
};


template <typename Tuple>
struct convolve_4 : public thrust::binary_function<Tuple,Tuple,Tuple>
{
  const Tuple d;
  const Tuple causal;
  convolve_4( Tuple _d, Tuple _causal ) : d(_d), causal(_causal) {}

  __host__ __device__
  Tuple operator()( const Tuple& x, const Tuple& y ) const
  {
    float sum = 0.0;
    for (unsigned int j = 0; j < 4; ++j)
      sum += thrust::get<j>(causal) * thrust::get<j>(x);
    
    for (unsigned int j = 0; j < 4; ++j)
      sum -= thrust::get<j>(d) * thrust::get<j>(y);
    
    return thrust::make_tuple(sum,
                              thrust::get<0>(y),
                              thrust::get<1>(y),
                              thrust::get<2>(y));
  }
};

void convolve_iir_gpu( float* input, float* output, int n_cols, int n_rows,
                       fastfilters::iir::Coefficients &coefs )
{
  cudaSetDevice(0);
  unsigned int n_pixels;
  cudaDeviceProp prop;
  thrust::tuple<float, float, float, float> d, causal, anticausal;

  n_pixels = n_cols * n_rows;

  for (int i=0; i<4;++i)
    {
      thrust::get<i>(d) = coefs.d[i]; 
      thrust::get<i>(causal) = coefs.n_causal[i];
      thrust::get<i>(anticausal) = coefs.n_anticausal[i];
    }

  thrust::device_vector<float> in(n_pixels+6);
  thrust::device_vector<float> out(n_pixels);
  
  thrust::fill(in.begin(), in.begin()+3, 0.0);
  thrust::copy(input, n_pixels, in.begin()+3);
  thrust::fill(in.begin()+3+n_pixels, in.end(), 0.0);

  typedef thrust::device_vector<float>::iterator FloatIterator;
  thrust::device_vector<thrust::tuple<float, float, float, float>> forth, back;
  
  thrust::exclusive_scan_by_key
    (thrust::make_zip_iterator
     (thrust::make_tuple(in.begin()+3, in.begin()+2,
                         in.begin()+1, in.begin())),
     thrust::make_zip_iterator
     (thrust::make_tuple(in.end()-3, in.end()-4,
                         in.end()-5, in.end()-6)),
     thrust::make_transform_iterator<int>
     (thrust::make_counting_iterator<int(0), row_idx(n_cols)),
     forth.begin(), // result
     thrust::make_tuple(0.0, 0.0, 0.0, 0.0),
     thrust::equal_to<int>,
     convolve_4(d, causal));

  thrust::reverse_iterator<FloatIterator> in_rev(in.end());
    
  thrust::exclusive_scan_by_key
    (thrust::make_zip_iterator
     (thrust::make_tuple(in_rev.begin()+3, in_rev.begin()+2,
                         in_rev.begin()+1, in_rev.begin())),
     thrust::make_zip_iterator
     (thrust::make_tuple(in_rev.end()-3, in_rev.end()-4,
                         in_rev.end()-5, in_rev.end()-6)),
     thrust::make_transform_iterator<int>
     (thrust::make_counting_iterator<int(0), row_idx(n_cols)),
     back.begin(), // result
     thrust::make_tuple(0.0, 0.0, 0.0, 0.0),
     thrust::equal_to<int>,
     convolve_4(d, anticausal));
    
  thrust::reverse_iterator<FloatIterator> back_rev(back.end());

  transform(forth.begin(), forth.end(), back_rev.begin(), out.begin(), addfirsts);
  thrust::copy(out.begin(), out.end(), output);
  
}


int main( int argc, char* argv[] )
{
  // check commandline parameters
  if( argc != 2 )
    {
      std::cout << "Usage: ./iirfilter-cuda pngfile" << std::endl;
      exit(0);
    }

  // determine file names
  std::string infile = argv[1];
  std::string outfile = infile.substr( 0, infile.length() - 4 ) + "_blurred_cuda.png";

  // initialize input parameters for function
  Image input = Image( infile );
  fastfilters::iir::Coefficients coefs( 5.0, 0 );
  float output_data[input.width()*input.height()];
  
  // time CUDA function
  auto begin = std::chrono::high_resolution_clock::now();
  int N = input.width() * input.height();
  size_t size = N * sizeof(float);
  cudaDeviceSynchronize();
  convolve_iir_gpu( input.data(), output_data, input.width(), input.height(), coefs);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  printf("%d", (end-begin).count()); // time in nanoseconds
  Image output = Image( output_data, input.width(), input.height() );
  output.write( outfile );

  return 0;
}
