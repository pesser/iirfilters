#include "image.hxx"
#include "coefficients.h"
#include "thrust_deriche.h"
#include "timer.h"
#include <iostream>

#include <thrust/device_vector.h>

template <class T>
void compute_thrust(std::vector<T>& output, Image& input,
                    const deriche_coeffs<T>& coeffs)
{
  int M = input.width();
  int N = input.height();
  int img_size = M * N;
  thrust::device_vector<T> d_input(img_size);
  thrust::copy(input.data(), input.data() + img_size, d_input.begin());

  thrust::device_vector<T> d_output(img_size);

  thrust::device_vector<T> buffer_l(img_size);
  thrust::device_vector<T> buffer_r(img_size);
  thrust::device_vector<T> buffer_m(img_size);

  deriche_thrust_2d<T>(
      coeffs,
      buffer_m.begin(),
      buffer_l.begin(), buffer_r.begin(),
      d_input.begin(), N, M, M, 1);

  cudaDeviceSynchronize();

  deriche_thrust_2d<T>(
      coeffs,
      d_output.begin(),
      buffer_l.begin(), buffer_r.begin(),
      buffer_m.begin(), M, N, 1, M);

  cudaDeviceSynchronize();
  output.resize(d_output.size());
  thrust::copy(d_output.begin(), d_output.end(), output.begin());
}

int main(int argc, char** argv)
{
  Timer<true> timer;
  float sigma = 5.0;
  int order = 4;

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

  std::vector<float> output;
  deriche_coeffs<float> c;
  deriche_precomp<float>(&c, sigma, order);

  timer.tick();
  compute_thrust(output, img, c);
  double time = timer.tock();

  std::cout << "Filtered in " << time << " ms" << std::endl;

  Image img_out(output.data(), img.width(), img.height());
  img_out.write(out_filename);

  return 0;
}
