/*
  Image class for tests of IIR filter functions
*/

#include <string>
#include <cmath>
#include <cassert>
#include <../png++/png.hpp>

#ifndef IMAGE_H
#define IMAGE_H

class Image
{

private:
  
  unsigned int _width, _height;
  float* _data;

public:

  Image( float* values, unsigned int w, unsigned int h );
  Image( std::string filename );

  unsigned int width()  { return _width;  }
  unsigned int height() { return _height; }
  const float* data()   { return _data;   }

  void write( std::string filename );

};


Image::Image( float* values, unsigned int w, unsigned int h )
{
  _data = values;
  _width = w;
  _height = h;
}

Image::Image( std::string filename )
{
  png::uint_32 x, y;

  // check image type
  assert( filename.length() > 3 );
  std::string ext = filename.substr( filename.length() - 3, filename.length() );
  assert( ext.compare("png") == 0 || ext.compare("PNG") == 0 );

  // read image
  png::image< png::gray_pixel > img( filename, png::require_color_space< png::gray_pixel >() );

  _width = img.get_width();
  _height = img.get_height();

  _data = new float[_width * _height];
  for( y = 0; y < _height; ++y)
    {
      for( x = 0; x < _width; ++x)
        {
          _data[y*_width+x] = (unsigned int) img.get_pixel( x, y ) / 255.0;
        }
    }
}

void Image::write( std::string filename )
{
  png::uint_32 x, y;
  unsigned int pixel_data;

  png::image< png::gray_pixel > out( _width, _height );
  for (y = 0; y < _height; ++y)
    {
      for (x = 0; x < _width; ++x)
        {
          pixel_data = (unsigned int) floor( _data[y*_width+x] * 255 );
          out.set_pixel( x, y, png::gray_pixel( pixel_data ) );
        }
    }
  out.write( filename );
}

#endif
