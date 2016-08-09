/*
  Program to compare convolve_iir_gpu function with sequential code on a random image of given size 

  Usage: ./program width height 
  
  Output (in ns): time_using_gpu time_using_cpu accuracy 
*/

#include "fastfilters.hxx"
#include "convolvegpu.hxx"

#include "../timer.h"
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

  Timer<true> timer;

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
  convolve_iir_gpu( input_data, output_data, w, h, coefs);
  double time = timer.tock();
  printf("%f ", time);

  timer.tick();
  fastfilters::iir::convolve_iir_inner_single_noavx(input_data, w, h, seq_output_data, coefs );
  time = timer.tock();
  printf("%f ", time);

  
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
