#include "image.hxx"
#include "coefficients.h"
#include "seq_deriche.h"
#include "timer.h"
#include <iostream>
#include <iomanip>

/** Blur image with sequential version. **/
int main(int argc, char** argv)
{
  Timer<false> timer;
  float sigma = 1.0;
  int order = 4;

  if(argc != 4)
  {
    std::cout <<
      "Useage: " << argv[0] << " <sigma> <in_png_file> <out_png_file>" << std::endl;
    return 1;
  }
  sigma = atof(argv[1]);
  const char* in_filename = argv[2];
  const char* out_filename = argv[3];

  Image img(in_filename);
  std::cout <<
    "Loaded image of size " << img.width() << "x" << img.height() << std::endl;

  int max_dim = std::max(img.width(), img.height());
  int img_size = img.width() * img.height();

  std::vector<float> output(img_size);
  deriche_coeffs<float> c;
  deriche_precomp<float>(&c, sigma, order);

  std::vector<float> buffer_l(max_dim);
  std::vector<float> buffer_r(max_dim);

  timer.tick();
  deriche_seq_2d<float>(
      c, output.data(), buffer_l.data(), buffer_r.data(),
      img.data(), img.height(), img.width());
  double time = timer.tock();

  std::cout << "Filtered with sigma=" << sigma << " in " << time << " s" << std::endl;

  Image img_out(output.data(), img.width(), img.height());
  img_out.write(out_filename);

  return 0;
}
