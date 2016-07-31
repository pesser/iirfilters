#include "coefficients.h"
#include "thrust_deriche.h"
#include "timer.h"
#include <iostream>

#include <thrust/device_vector.h>

template <class T>
void compute_thrust(std::vector<T>& output, const std::vector<T>& input,
                    const deriche_coeffs<T>& coeffs, int N)
{
  Timer<true> timer;

  timer.tick();
  thrust::device_vector<T> d_input(input);
  thrust::device_vector<T> d_output(input.size());

  thrust::device_vector<T> buffer_l(input.size());
  thrust::device_vector<T> buffer_r(input.size());
  thrust::device_vector<T> buffer_m(input.size());
  double time_htd = timer.tock();

  timer.tick();
  deriche_thrust_2d<T>(
      coeffs,
      buffer_m.begin(),
      buffer_l.begin(), buffer_r.begin(),
      d_input.begin(), N, N, N, 1);
  double time_horizontal = timer.tock();

  cudaDeviceSynchronize();

  timer.tick();
  deriche_thrust_2d<T>(
      coeffs,
      d_output.begin(),
      buffer_l.begin(), buffer_r.begin(),
      buffer_m.begin(), N, N, 1, N);
  double time_vertical = timer.tock();

  timer.tick();
  output.resize(d_output.size()); // output should have the right capacity
                                  // already, otherwise this can be quite slow
  cudaDeviceSynchronize();
  thrust::copy(d_output.begin(), d_output.end(), output.begin());
  double time_dth = timer.tock();

  std::cout << "HostDevice: " << time_htd << std::endl;
  std::cout << "Horizontal: " << time_horizontal << std::endl;
  std::cout << "Vertical: " << time_vertical << std::endl;
  std::cout << "DeviceHost: " << time_dth << std::endl;
}

int main(int argc, char** argv)
{
  typedef float prec_t;

  Timer<true> timer;
  float sigma = 5.0;
  int order = 4;

  if(argc != 2)
  {
    std::cout <<
      "Useage: " << argv[0] << " <n_rows>" << std::endl;
    return 1;
  }
  int N = atoi(argv[1]);
  std::vector<prec_t> input_data(N * N, 1.0f);

  std::vector<prec_t> output(N * N);
  deriche_coeffs<prec_t> c;
  deriche_precomp<prec_t>(&c, sigma, order);

  timer.tick();
  compute_thrust(output, input_data, c, N);
  double time = timer.tock();

  std::cout << "Total " << time << std::endl;

  return 0;
}
