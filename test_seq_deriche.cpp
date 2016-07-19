#include "coefficients.h"
#include "seq_deriche.h"
#include <iostream>
#include <iomanip>

int main()
{
  float sigma = 1.0;
  int order = 3;

  std::vector<float> input{
    0.0, 0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 0.9, 0.6, 0.4, 0.2, 0.1, 0.0};
  std::vector<float> output(input.size());
  std::vector<float> buffer_l(input.size());
  std::vector<float> buffer_r(input.size());

  deriche_coeffs<float> c;
  deriche_precomp<float>(&c, sigma, order);
  deriche_seq<float>(c, output.data(), buffer_l.data(), buffer_r.data(),
                     input.data(), input.size());

  unsigned int w = 8;
  std::cout << std::setprecision(6);
  std::cout << "Input:" << std::endl;
  for(unsigned int i = 0; i < input.size(); ++i)
    std::cout << std::setw(w) << input[i] << " ";
  std::cout << std::endl << "Output:" << std::endl;
  for(unsigned int i = 0; i < input.size(); ++i)
    std::cout << std::setw(w) << output[i] << " ";
  std::cout << std::endl;

  return 0;
}
