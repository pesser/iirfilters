#include "coefficients.h"
#include "thrust_deriche.h"
#include "timer.h"
#include <iostream>

#include <thrust/device_vector.h>

template <class T>
void compute_thrust(std::vector<T>& output, const std::vector<T>& input,
                    const deriche_coeffs<T>& coeffs, int N)
{
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  cublasHandle_t handle;
  cublasSafeCall(cublasCreate(&handle));

  Timer<true> timer;

  timer.tick();
  thrust::device_vector<T> d_input(input);
  thrust::device_vector<T> d_output(input.size());

  thrust::device_vector<T> buffer_l(input.size());
  thrust::device_vector<T> buffer_r(input.size());
  double time_htd = timer.tock();


  timer.tick();
  deriche_thrust_2d<T>(coeffs,
                       d_output.begin(),
                       buffer_l.begin(), buffer_r.begin(),
                       d_input.begin(), N, N,
                       s1, s2, &handle);
  double time_vertical = timer.tock();

  timer.tick();
  output.resize(d_output.size()); // output should have the right capacity
                                  // already, otherwise this can be quite slow
  thrust::copy(d_output.begin(), d_output.end(), output.begin());
  double time_dth = timer.tock();

  std::cout << N << ",";
  std::cout << time_htd << ",";
  std::cout << 0 << ",";
  std::cout << time_vertical << ",";
  std::cout << time_dth;
  std::cout << std::endl;

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cublasSafeCall(cublasDestroy(handle));
}

// produce csv timings for N, PreInit, Horizontal, Vertical, Postinit
int main(int argc, char** argv)
{
  typedef float prec_t;

  float sigma = 5.0;
  int order = 4;

  if(argc != 2)
  {
    std::cout <<
      "Usage: " << argv[0] << " <n_rows>" << std::endl;
    return 1;
  }
  int N = atoi(argv[1]);
  std::vector<prec_t> input_data(N * N, 1.0f);

  std::vector<prec_t> output(N * N);
  deriche_coeffs<prec_t> c;
  deriche_precomp<prec_t>(&c, sigma, order);

  compute_thrust(output, input_data, c, N);

  return 0;
}
