/*
  Program to time Sven Peter's convolve_iir_inner_single_noavx function
  Prints time in nanoseconds

  Usage: ./iir-sequential pngfile
 */

#include "fastfilters.hxx"
#include "image.hxx"

#include <iostream>
#include <string>
#include <chrono>
#include <stdlib.h>
  
int main( int argc, char* argv[] )
{
  // check commandline parameters
  if( argc != 2 )
    {
      std::cout << "Usage: ./iir-sequential pngfile" << std::endl;
      exit(0);
    }
      
  // determine file names 
  std::string infile = argv[1]; 
  std::string outfile = infile.substr( 0, infile.length() - 4 ) + "_blurred_seq.png";
  
   // initialize input parameters for function
  Image input = Image(infile);  
  fastfilters::iir::Coefficients coefs( 5.0, 0 );
  float output_data[input.width()*input.height()];
  
  // time function
  auto begin = std::chrono::high_resolution_clock::now();
  fastfilters::iir::convolve_iir_inner_single_noavx(input.data(), input.width(),
                                                    input.height(), output_data, coefs );
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << std::endl;

  // write result back to output file
  Image output = Image( output_data, input.width(), input.height() );
  output.write( outfile );

  return 0;
}
