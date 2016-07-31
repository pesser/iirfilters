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

#include <thrust/functional.h>
#include <thrust/tabulate.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>

typedef thrust::tuple<float, float, float, float> FloatTuple;


struct RandGen
{
    RandGen() {}

    __host__ __device__
    float operator () (int idx)
    { 
       thrust::default_random_engine randEng;
       thrust::uniform_real_distribution<float> uniDist(-3.2, 3.2);
       randEng.discard(idx);
       return uniDist(randEng);
    }
};

struct convolve_4
{
  const FloatTuple d;
  const FloatTuple causal;

  convolve_4( const FloatTuple _d, const FloatTuple _causal ) : d(_d), causal(_causal) {}

   template <typename Tuple, typename Tuple2>
  __host__ __device__
  Tuple operator()(const Tuple &y, const Tuple2 &x)
  {
    using thrust::get;
    float sum =  get<0>(causal) * get<0>(x)
               + get<1>(causal) * get<1>(x)
               + get<2>(causal) * get<2>(x)
               + get<3>(causal) * get<3>(x)
               - get<0>(d) * get<0>(y) 
               - get<1>(d) * get<1>(y)
               - get<2>(d) * get<2>(y) 
               - get<3>(d) * get<3>(y);
    return thrust::make_tuple(sum, get<0>(y), get<1>(y), get<2>(y));
  }
};

struct addfirsts : public thrust::binary_function<FloatTuple, FloatTuple, float>
{
  __host__ __device__
  float operator()( const FloatTuple& x, const FloatTuple& y ) const
  {
    return thrust::get<0>(x) + thrust::get<0>(y);
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

void convolve_iir_gpu(const float* input, float* output, int n_cols, int n_rows,
                       fastfilters::iir::Coefficients &coefs )
{
  cudaSetDevice(0);
  unsigned int n_pixels, n_border;

  n_pixels = n_cols * n_rows;
  n_border = 3; // since window size = 4

  FloatTuple d(coefs.d[0],coefs.d[1], coefs.d[2], coefs.d[3]);
  FloatTuple causal(coefs.n_causal[0],coefs.n_causal[1], coefs.n_causal[2], coefs.n_causal[3]);
  FloatTuple anticausal(coefs.n_anticausal[0],coefs.n_anticausal[1], coefs.n_anticausal[2], coefs.n_anticausal[3]);

  thrust::device_vector<float> in(n_pixels+2*n_border); // left and right border 
  thrust::device_vector<float> out(n_pixels);
  
  // pad border with zeros
  thrust::fill(in.begin(), in.begin()+n_border, 0.0);
  thrust::copy_n(input, n_pixels, in.begin()+n_border);
  thrust::fill(in.begin()+n_border+n_pixels, in.end(), 0.0);

  FloatTuple zero(0,0,0,0);
  thrust::device_vector<FloatTuple> forth(n_pixels,zero);
  thrust::device_vector<FloatTuple> back(n_pixels,zero);

  thrust::device_vector<int> key(n_pixels);
  thrust::tabulate(key.begin(), key.end(), row_idx(n_cols));
  thrust::equal_to<int> eq;

  exclusive_scan_by_key(key.begin(), 
                        key.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(in.begin()+3, in.begin()+2, in.begin()+1, in.begin()))+1,
                        forth.begin(), 
                        thrust::make_tuple(coefs.d[0]*in[3],0,0,0),
                        eq,
                        convolve_4(d,causal));
  
  auto in_rev = in.rbegin();
  thrust::exclusive_scan_by_key
    (key.begin(),
     key.end(),
     thrust::make_zip_iterator (thrust::make_tuple(in_rev+3, in_rev+2, in_rev+1, in_rev)),
     back.begin(),
     thrust::make_tuple(coefs.d[0]*in_rev[3], 0.0, 0.0, 0.0),
     eq,
     convolve_4(d, anticausal));
    
  auto back_rev = back.rbegin();
  thrust::transform(forth.begin(), forth.end(), back_rev, out.begin(), addfirsts());
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

  // for testing - TODO!
  int n = 10;
  thrust::device_vector<float> d_t1(n);
  thrust::tabulate(d_t1.begin(), d_t1.end(), RandGen());
  
  // determine file names
  std::string infile = argv[1];
  std::string outfile = infile.substr( 0, infile.length() - 4 ) + "_blurred_thrust.png";

  // initialize input parameters for function
  Image input = Image( infile );
  fastfilters::iir::Coefficients coefs( 5.0, 0 );
  float output_data[input.width()*input.height()];
  
  // time CUDA function
  auto begin = std::chrono::high_resolution_clock::now();
  cudaDeviceSynchronize();
  convolve_iir_gpu( input.data(), output_data, input.width(), input.height(), coefs);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  printf("%d", (end-begin).count()); // time in nanoseconds
  Image output = Image( output_data, input.width(), input.height() );
  output.write( outfile );

  return 0;
}
