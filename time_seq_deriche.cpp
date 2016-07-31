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
  std::vector<T> buffer_m(N * N);
  double time_init = timer.tock();

  timer.tick();
  deriche_seq_2d<T>(
      coeffs,
      buffer_m.data(),
      buffer_l.data(), buffer_r.data(),
      input.data(), N, N, N, 1);
  double time_horizontal = timer.tock();

  timer.tick();
  deriche_seq_2d<T>(
      coeffs,
      output.data(),
      buffer_l.data(), buffer_r.data(),
      buffer_m.data(), N, N, 1, N);
  double time_vertical = timer.tock();

  std::cout << "Init: " << time_init << std::endl;
  std::cout << "Horizontal: " << time_horizontal << std::endl;
  std::cout << "Vertical: " << time_vertical << std::endl;
}

int main(int argc, char** argv)
{
  typedef float prec_t;

  Timer<false> timer;
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
  compute_seq(output, input_data, c, N);
  double time = timer.tock();

  std::cout << "Total: " << time << std::endl;

  return 0;
}
