#include "coefficients.h"
#include "thrust_deriche.h"
#include "timer.h"
#include <iostream>

#include <thrust/device_vector.h>

struct transpose_index : public thrust::unary_function<int, int>
{
  int m, n;

  __host__ __device__
    transpose_index(int m, int n) : m(m), n(n) {}

  __host__ __device__
    int operator()(int linear_index)
    {
      int i = linear_index / n;
      int j = linear_index % n;

      return m * j + i;
    }
};

template <class T>
void transpose(int m, int n, thrust::device_vector<T>& src, thrust::device_vector<T>& dst)
{
	thrust::counting_iterator<int> indices(0);

	thrust::gather
		(thrust::make_transform_iterator(indices, transpose_index(n, m)),
		 thrust::make_transform_iterator(indices, transpose_index(n, m)) + dst.size(),
		 src.begin(),
		 dst.begin());
}

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

#ifdef _NOTRANSPOSE
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
#else
  timer.tick();
  deriche_thrust_2d<T>(
      coeffs,
      buffer_m.begin(),
      buffer_l.begin(), buffer_r.begin(),
      d_input.begin(), N, N, 1, N);
  transpose(N, N, buffer_m, d_output);
  double time_horizontal = timer.tock();

  cudaDeviceSynchronize();

  timer.tick();
  deriche_thrust_2d<T>(
      coeffs,
      buffer_m.begin(),
      buffer_l.begin(), buffer_r.begin(),
      d_output.begin(), N, N, 1, N);
  transpose(N, N, buffer_m, d_output);
  double time_vertical = timer.tock();
#endif

  timer.tick();
  output.resize(d_output.size()); // output should have the right capacity
                                  // already, otherwise this can be quite slow
  cudaDeviceSynchronize();
  thrust::copy(d_output.begin(), d_output.end(), output.begin());
  double time_dth = timer.tock();

  std::cout << N << ",";
  std::cout << time_htd << ",";
  std::cout << time_horizontal << ",";
  std::cout << time_vertical << ",";
  std::cout << time_dth;
  std::cout << std::endl;
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
      "Useage: " << argv[0] << " <n_rows>" << std::endl;
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
