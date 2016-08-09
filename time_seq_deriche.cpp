#include "coefficients.h"
#include "seq_deriche.h"
#include "timer.h"
#include <iostream>
#include <iomanip>

template <class T>
void compute_seq(std::vector<T>& output, const std::vector<T>& input,
                 const deriche_coeffs<T>& coeffs, int N)
{
  Timer<false> timer;
  timer.tick();
  std::vector<T> buffer_l(N);
  std::vector<T> buffer_r(N);
  double time_init = timer.tock();

  timer.tick();
  deriche_seq_2d<float>(
      coeffs, output.data(), buffer_l.data(), buffer_r.data(),
      input.data(), N, N);
  double time_vertical = timer.tock();

  std::cout << N << ",";
  std::cout << time_init << ",";
  std::cout << 0.0 << ",";
  std::cout << time_vertical << ",";
  std::cout << 0.0;
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
      "Usage: " << argv[0] << " <n_rows>" << std::endl;
    return 1;
  }
  int N = atoi(argv[1]);
  std::vector<prec_t> input_data(N * N, 1.0f);

  std::vector<prec_t> output(N * N);
  deriche_coeffs<prec_t> c;
  deriche_precomp<prec_t>(&c, sigma, order);

  compute_seq(output, input_data, c, N);

  return 0;
}
