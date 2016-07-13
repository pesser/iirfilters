/*
  Program to time Sven Peter's convolve_iir_inner_single_noavx function
  Prints time in nanoseconds

  Usage: ./iir-sequential pngfile
 */

#include "fastfilters.hxx"

#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <vector>
#include <iomanip>
#include <chrono>
#include <stdlib.h>
#include <png++/png.hpp>


int time_png( std::string filename )
{
  unsigned int w, h;
  png::uint_32 x, y;

  // check image type
  assert( filename.length() > 3 );
  std::string ext = filename.substr( filename.length() - 3, filename.length() );
  assert( ext.compare("png") == 0 || ext.compare("PNG") == 0 );

  // determine name of output file
  std::string outfile = filename.substr( 0, filename.length() - 4 ) + "_blurred_seq.png";

  // read image
  png::image< png::gray_pixel > img( filename, png::require_color_space< png::gray_pixel >() );
  w = img.get_width();
  h = img.get_height();

  // initialize input parameters for function
  std::vector<float> input( w * h );
  for( y=0; y<h; ++y)
    {
      for( x=0; x<w; ++x)
        {
          input[y*w+x] = (unsigned int) img.get_pixel( x, y ) / 255.0;
        }
    }

  std::vector<float> output( w * h );
  fastfilters::iir::Coefficients coefs( 5.0, 0 );

  // time function
  auto begin = std::chrono::high_resolution_clock::now();
  fastfilters::iir::convolve_iir_inner_single_noavx( input.data(), w, h, output.data(), coefs );
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << std::endl;

  // write result back to output file
  png::image< png::gray_pixel > out( w, h );

  for (y = 0; y < h; ++y)
    {
      for (x = 0; x < w; ++x)
        {
          out.set_pixel(x, y, png::gray_pixel( (unsigned int) floor(output[y*w+x] * 255) ) );
        }
    }
  out.write( outfile );
  
  return 0;
}

int main( int argc, char* argv[] )
{
  if( argc != 2 )
    {
      std::cout << "Usage: ./iir-sequential pngfile" << std::endl;
      exit(0);
    }
      
  std::cout << std::setprecision( 30 );

  std::string filename = argv[1]; 
  time_png( filename );

  return 0;
}
