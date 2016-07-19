#include "image.hxx"
#include "coefficients.h"
#include "seq_deriche.h"
#include "timer.h"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv)
{
  Timer<false> timer;
  float sigma = 5.0;
  int order = 3;

  if(argc != 3)
  {
    std::cout <<
      "Useage: " << argv[0] << " <in_png_file> <out_png_file>" << std::endl;
    return 1;
  }
  const char* in_filename = argv[1];
  const char* out_filename = argv[2];

  Image img(in_filename);
  std::cout <<
    "Loaded image of size " << img.width() << "x" << img.height() << std::endl;

  int max_dim = std::max(img.width(), img.height());
  int img_size = img.width() * img.height();

  std::vector<float> output(img_size);

  std::vector<float> buffer_l(max_dim);
  std::vector<float> buffer_r(max_dim);
  std::vector<float> buffer_m(img_size);

  deriche_coeffs<float> c;
  deriche_precomp<float>(&c, sigma, order);

  timer.tick();
  deriche_seq_2d<float>(
      c, output.data(), buffer_l.data(), buffer_r.data(), buffer_m.data(),
      img.data(), img.width(), img.height());
  double time = timer.tock();

  std::cout << time << std::endl;

  Image img_out(output.data(), img.width(), img.height());
  img_out.write(out_filename);

  return 0;
}
