/*
  Program to compare convolve_iir_gpu function with sequential code on a random image of given size 

  Usage: ./program width height 
  
  Output (in s): time_using_gpu time_using_cpu 
*/

#include "fastfilters.hxx"
#include "convolvegpu.hxx"
#include "timer.h"

#include <chrono>
#include <stdlib.h>
#include <stdio.h>

int main( int argc, char* argv[] )
{
  // check commandline parameters
  if( argc != 2 )
    {
      printf("Usage: ./program N\n");
      exit(0);
    }
  Timer<true> timer;  

  // initialize input parameters
  int N = atoi(argv[1]);
  int w = N;
  int h = N;
  
  printf("%d,", N);

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
  timer.tick();
  convolve_iir_gpu( input_data, output_data, w, h, coefs );
  double par_time = timer.tock();
  printf("%f,", par_time);

  timer.tick();
  fastfilters::iir::convolve_iir_inner_single_noavx(input_data, w, h, seq_output_data, coefs );
  double seq_time = timer.tock();
  printf("%f", seq_time);
  
  return 0;
}
