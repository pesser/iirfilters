/*
  Program to compare convolve_iir_gpu function with sequential code on a random image of given size 

  Usage: ./program width height 
  
  Output (in ns): time_using_gpu time_using_cpu accuracy 
*/

#include "fastfilters.hxx"

#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


int main( int argc, char* argv[] )
{
  // check commandline parameters
  if( argc != 3 )
    {
      std::cout << "Usage: ./program width height" << std::endl;
      exit(0);
    }
  
  // initialize input parameters
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[1]);
   
  float input_data[w*h];
  float output_data[w*h];
  float seq_output_data[w*h];

  // create random image
  srand( (unsigned) time(NULL) );
  for( int i=0; i<w*h; ++i )
    {
      input_data[i] = rand()/float(RAND_MAX);
    }
  
  // time functions
  auto begin = std::chrono::high_resolution_clock::now();
  cudaDeviceSynchronize();
  convolve_iir_gpu( input_data, output_data, w, h, coefs);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  printf("%d ", (end-begin).count());

  begin = std::chrono::high_resolution_clock::now();
  cudaDeviceSynchronize();
  convolve_iir_gpu( input_data2, output_data2, w, h, coefs);
  fastfilters::iir::convolve_iir_inner_single_noavx(input_data, w, h, output_data, coefs );
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  printf("%d ", (end-begin).count());

  
  // calculate RMS error
  float rmse = 0.0;
  for( int i=0; i<w*h; ++i )
    rmse += pow( seq_output_data[i] - output_data[i], 2);
  rmse = sqrt(rmse);
  printf("%f", rmse);
  
  return 0;
}
