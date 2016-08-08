#pragma once

#include <iostream>
#include <iomanip>
#include <cmath>

void fail()
{
  std::cerr << "\033[1;31m******** Test failed. ********\033[0m\033[0m" << std::endl;
  exit(1);
}

template <class It>
void print(It begin, It end, int linebreak = 0)
{
  auto prec = std::cout.precision();
  std::cout.precision(4);

  int i = 1;
  while(begin != end)
  {
    std::cout << std::setw(8) << *begin;
    ++begin;
    if(linebreak > 0 && i++ % linebreak == 0)
      std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout.precision(prec);
}
  

template <class T>
void assert_same(const std::vector<T>& a, const std::vector<T>& b,
                 const double TOL = 1e-4)
{
  if(a.size() != b.size())
    fail();
  for(unsigned int i = 0; i < a.size(); ++i)
  {
    if(std::abs(a[i] - b[i]) > TOL)
    {
      fail();
    }
  }
}
