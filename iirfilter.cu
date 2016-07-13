/*
  Program to time functions convolving images using cuda
  Prints time in nanoseconds

  Usage: ./iirfilter-cuda pngfile
*/

#include "fastfilters.hxx"
#include "image.hxx"

#include <iostream>
#include <string>
#include <chrono>
#include <stdlib.h>

__global__ void convolve_row( float* input, float* output, int N, const Coefficients &coefs)
{
  int dim = threadIdx.x;
    
  const float *cur_line = input + dim * N;
  float *cur_output = output + dim * N;
  float xtmp[4];
  float ytmp[4];
  float tmpbfr[N];

  for (unsigned int i = 0; i < 4; ++i)
    xtmp[i] = ytmp[i] = 0.0;

  // left border
  for (unsigned int i = 0; i < coefs.n_border; ++i) {
    float sum = 0.0;

    xtmp[0] = cur_line[coefs.n_border - i];
    for (unsigned int j = 0; j < 4; ++j)
      sum += coefs.n_causal[j] * xtmp[j];
    for (unsigned int j = 0; j < 4; ++j)
      sum -= coefs.d[j] * ytmp[j];
    for (unsigned int j = 3; j > 0; --j) {
      xtmp[j] = xtmp[j - 1];
      ytmp[j] = ytmp[j - 1];
    }

    ytmp[0] = sum;
  }

  // causal pass
  for (unsigned int i = 0; i < N; ++i) {
    float sum = 0.0;

    xtmp[0] = cur_line[i];
    for (unsigned int j = 0; j < 4; ++j)
      sum += coefs.n_causal[j] * xtmp[j];
    for (unsigned int j = 0; j < 4; ++j)
      sum -= coefs.d[j] * ytmp[j];
    for (unsigned int j = 3; j > 0; --j) {
      xtmp[j] = xtmp[j - 1];
      ytmp[j] = ytmp[j - 1];
    }

    tmpbfr[i] = sum;
    ytmp[0] = sum;
  }

  // reset variables for anti-causal pass
  for (unsigned int i = 0; i < 4; ++i)
    xtmp[i] = ytmp[i] = 0.0;

  // right border
  for (int i = coefs.n_border; i > 0; --i) {
    float sum = 0.0;

    for (unsigned int j = 0; j < 4; ++j)
      sum += coefs.n_anticausal[j] * xtmp[j];
    for (unsigned int j = 0; j < 4; ++j)
      sum -= coefs.d[j] * ytmp[j];
    for (unsigned int j = 3; j > 0; --j) {
      xtmp[j] = xtmp[j - 1];
      ytmp[j] = ytmp[j - 1];
    }

    xtmp[0] = cur_line[N - i];
    ytmp[0] = sum;
  }

  // anti-causal pass
  for (int i = N - 1; i >= 0; --i) {
    float sum = 0.0;

    for (unsigned int j = 0; j < 4; ++j)
      sum += coefs.n_anticausal[j] * xtmp[j];
    for (unsigned int j = 0; j < 4; ++j)
      sum -= coefs.d[j] * ytmp[j];
    for (unsigned int j = 3; j > 0; --j) {
      xtmp[j] = xtmp[j - 1];
      ytmp[j] = ytmp[j - 1];
    }

    xtmp[0] = cur_line[i];
    ytmp[0] = sum;
    cur_output[i] = tmpbfr[i] + sum;
  }
}


void convolve_iir_gpu( float* input, float* output, int n_cols, int n_rows,
                       const Coefficients &coefs)
{
  convolve_row<< 1, n_rows >>(input, output, n_cols);
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
  Image input = Image(infile);
  fastfilters::iir::Coefficients coefs( 5.0, 0 );
  float output_data[input.width()*input.height()];

  // time CUDA function
  auto begin = std::chrono::high_resolution_clock::now();
  int N = input.width() * input.height();
  size_t size = N * sizeof(float);
  cudaDeviceSynchronize();
  float* d_in;
  cudaMalloc( &d_in, size );
  float* d_out;
  cudaMalloc( &d_out, size );
  cudaMemcpy( d_in, input.data(), size );
  convolve_iir_gpu( d_in, d_out, input.width(), input.height(), coefs);
  cudaMemcpy( output_data, d_out, size );
  cudaFree(d_in);
  cudaFree(d_out);
  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << std::endl;

  // write result back to output file
  Image output = Image( output_data, input.width(), input.height() );
  output.write( outfile );

  return 0;
}
