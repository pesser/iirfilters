/*
  Program to time functions convolving images using cuda  
  Prints time in nanoseconds

  Usage: ./iir-cuda pngfile
 */

#include "fastfilters.hxx"
#include "image.hxx"

#include <iostream>
#include <string>
#include <chrono>
#include <stdlib.h>


void convolve_iir_gpu(thrust::device_vector<float>& input,
                      thrust::device_vector<float>& output,
                      const Coefficients &coefs)
{
  // TODO
}


int main( int argc, char* argv[] )
{
  // check commandline parameters
  if( argc != 2 )
    {
      std::cout << "Usage: ./iir-cuda pngfile" << std::endl;
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
  cudaDeviceSynchronize();
  thrust::device_vector<float> in(input.data(), input.width()*input.height());
  thrust::device_vector<float> out(input.width()*input.height());
  convolve_iir_gpu(in, out, coefs);
  thrust::copy(out.begin(), out.end(), thrust::device_ptr_cast(output_data));
  cudaDeviceSynchronize();
  
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << std::endl;

  // write result back to output file
  Image output = Image( output_data, input.width(), input.height() );
  output.write( outfile );

  return 0;
}
