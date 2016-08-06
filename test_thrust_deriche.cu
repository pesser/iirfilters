#include "coefficients.h"
#include "seq_deriche.h"
#include "thrust_deriche.h"
#include "utils.h"
#include "image.hxx"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <class T>
void compute_thrust(std::vector<T>& output, Image& input,
                    const deriche_coeffs<T>& coeffs)
{
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  cublasHandle_t handle;
  cublasSafeCall(cublasCreate(&handle));

  unsigned int size = input.width() * input.height();
  thrust::device_vector<T> d_input(input.data(), input.data() + size);
  thrust::device_vector<T> d_output(size);

  thrust::device_vector<T> buffer_l(size);
  thrust::device_vector<T> buffer_r(size);

  deriche_thrust_2d<T>(coeffs,
                       d_output.begin(),
                       buffer_l.begin(), buffer_r.begin(),
                       d_input.begin(), input.height(), input.width(),
                       s1, s2, &handle);
  cudaDeviceSynchronize();

  output.resize(d_output.size());
  thrust::copy(d_output.begin(), d_output.end(), output.begin());

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cublasSafeCall(cublasDestroy(handle));
}

template <class T>
void compute_seq(std::vector<T>& output, Image& input,
                 const deriche_coeffs<T>& coeffs)
{
  output.resize(input.width() * input.height());
  int max_dim = std::max(input.width(), input.height());
  std::vector<T> buffer_l(max_dim);
  std::vector<T> buffer_r(max_dim);
  deriche_seq_2d<float>(
      coeffs, output.data(), buffer_l.data(), buffer_r.data(),
      input.data(), input.height(), input.width());
}

// compare sequential and thrust version on image
int main()
{
  float sigma = 4.0;
  int order = 4;

  deriche_coeffs<float> c;
  deriche_precomp<float>(&c, sigma, order);

  Image img("images/alice_fly_bubble.png");

  std::vector<float> seq_result, thrust_result;

  compute_seq(seq_result, img, c);
  compute_thrust(thrust_result, img, c);

  assert_same(seq_result, thrust_result);

  return 0;
}
