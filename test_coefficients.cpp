#include "coefficients.h"
#include <iostream>

int main()
{
  float sigma = 6.2;
  int order = 3;
  float a_reference[] = {8.82565e-39, -2.30108, 1.79901};

  deriche_coeffs<float> c;
  deriche_precomp<float>(&c, sigma, order);

  for(int i = 0; i < c.K; ++i)
  {
    std::cout << c.a[i] << std::endl;
    if(std::fabs(c.a[i] - a_reference[i]) > 1e-5)
    {
      std::cerr << "WARNING: Regression test failed." << std::endl;
      exit(1);
    }
  }

  return 0;
}
