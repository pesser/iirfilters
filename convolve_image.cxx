/*
  Program to test convolve_iir_gpu function on a given image 
  Prints time in nanoseconds

  Usage: ./program pngfile
*/

#include "fastfilters.hxx"
#include "image.hxx"

#include <string>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>


int main( int argc, char* argv[] )
{
  // check commandline parameters
  if( argc != 2 )
    {
      std::cout << "Usage: ./program pngfile" << std::endl;
      exit(0);
    }

  // determine file names
  std::string infile = argv[1];
  std::string outfile = infile.substr( 0, infile.length() - 4 ) + "_blurred_thrust.png";

  // initialize input parameters for function
  Image input = Image( infile );
  fastfilters::iir::Coefficients coefs( 5.0, 0 );
  float output_data[input.width()*input.height()];

  // time function
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
