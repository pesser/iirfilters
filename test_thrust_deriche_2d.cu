#include "coefficients.h"
#include "seq_deriche.h"
#include "thrust_deriche.h"
#include "utils.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

const unsigned int N = 30; // height
const unsigned int M = 20; // width

template <class T>
void compute_thrust(std::vector<T>& output, const std::vector<T>& input,
                    const deriche_coeffs<T>& coeffs)
{
  assert(input.size() == N*M);
  thrust::device_vector<T> d_input(input);
  thrust::device_vector<T> d_output(input.size());

  thrust::device_vector<T> buffer_l(input.size());
  thrust::device_vector<T> buffer_r(input.size());
  thrust::device_vector<T> buffer_m(input.size());

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

  output.resize(d_output.size());
  cudaDeviceSynchronize();
  thrust::copy(d_output.begin(), d_output.end(), output.begin());
}

template <class T>
void compute_seq(std::vector<T>& output, const std::vector<T>& input,
                 const deriche_coeffs<T>& coeffs)
{
  assert(N*M == input.size());
  output.resize(input.size());
  std::vector<T> buffer_l(std::max(N, M));
  std::vector<T> buffer_r(std::max(N, M));
  std::vector<T> buffer_m(input.size());

  deriche_seq_2d<T>(
      coeffs,
      output.data(),
      buffer_l.data(), buffer_r.data(),
      buffer_m.data(),
      input.data(), M, N);
}

// compare sequential and thrust version on constant matrix
int main()
{
  typedef float prec_t;
  prec_t sigma = 1.0;
  int order = 4;

  deriche_coeffs<prec_t> c;
  deriche_precomp<prec_t>(&c, sigma, order);

  std::vector<prec_t> input_data(N * M, 1.0f);

  std::vector<prec_t> seq_result, thrust_result;

  compute_seq(seq_result, input_data, c);
  compute_thrust(thrust_result, input_data, c);

  print(seq_result.begin(), seq_result.end(), M);
  print(thrust_result.begin(), thrust_result.end(), M);
  assert_same(seq_result, thrust_result);

  return 0;
}
