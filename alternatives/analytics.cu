/*
  Program to compare convolve_iir_gpu function with sequential code on a random image of given size 

  Usage: ./program width height 
  
  Output (in ns): time_using_gpu time_using_cpu accuracy 
*/

#include "fastfilters.hxx"
#include "convolvegpu.hxx"

#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


int main( int argc, char* argv[] )
{
  // check commandline parameters
  if( argc != 3 )
    {
      printf("Usage: ./program width height\n");
      exit(0);
    }
  
  // initialize input parameters
  int w = atoi(argv[1]);
  int h = atoi(argv[2]);

  float* input_data;
  float* output_data;
  float* seq_output_data;

  input_data = (float*) malloc(w*h*sizeof(float));
  output_data = (float*) malloc(w*h*sizeof(float));
  seq_output_data = (float*) malloc(w*h*sizeof(float));

  // create random image
  srand( (unsigned) time(NULL) );
  for( int i=0; i<w*h; ++i )
    {
      input_data[i] = rand()/float(RAND_MAX);
    }

  // get coefficients for convolution
  fastfilters::iir::Coefficients coefs( 5.0, 0 );
  
  // time functions
  cudaDeviceSynchronize();
  auto begin = std::chrono::high_resolution_clock::now();
  convolve_iir_gpu( input_data, output_data, w, h, coefs);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  printf("%d ", (end-begin).count());

  begin = std::chrono::high_resolution_clock::now();
  fastfilters::iir::convolve_iir_inner_single_noavx(input_data, w, h, seq_output_data, coefs );
  end = std::chrono::high_resolution_clock::now();
  printf("%d ", (end-begin).count());

  
  // calculate RMS error
  long double rmse = 0.0;
  for( int i=0; i<w*h; ++i ) 
    {
      rmse += pow( seq_output_data[i] - output_data[i], 2)/(w*h);
    }
  printf("%Lf ", rmse);
  rmse = sqrt(rmse);
  printf("%Lf", rmse);
  
  return 0;
}
