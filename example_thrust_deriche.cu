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
  Timer<true> timer;
  double time_total = 0.0;
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  cublasHandle_t handle;
  cublasSafeCall(cublasCreate(&handle));

  timer.tick();
  unsigned int size = input.width() * input.height();
  thrust::device_vector<T> d_input(input.data(), input.data() + size);
  thrust::device_vector<T> d_output(size);

  thrust::device_vector<T> buffer_l(size);
  thrust::device_vector<T> buffer_r(size);

  time_total += timer.tock();

  timer.tick();
  deriche_thrust_2d<T>(coeffs,
                       d_output.begin(),
                       buffer_l.begin(), buffer_r.begin(),
                       d_input.begin(), input.height(), input.width(),
                       s1, s2, &handle);
  double time_compute = timer.tock();
  time_total += time_compute;

  timer.tick();
  output.resize(d_output.size());
  thrust::copy(d_output.begin(), d_output.end(), output.begin());
  time_total += timer.tock();

  std::cout << "Filtered in " << time_compute << " s (" << time_total << ")" << std::endl;

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cublasSafeCall(cublasDestroy(handle));
}

/** Blur image with thrust version. **/
int main(int argc, char** argv)
{
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

  std::vector<float> output;
  deriche_coeffs<float> c;
  deriche_precomp<float>(&c, sigma, order);

  compute_thrust(output, img, c);


  Image img_out(output.data(), img.width(), img.height());
  img_out.write(out_filename);

  return 0;
}
