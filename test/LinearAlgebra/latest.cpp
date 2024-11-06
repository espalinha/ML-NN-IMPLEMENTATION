#include "../../LA/LinearAlgebra.hpp"
#include <iostream>

int main() {
  double x, y;
  x = 10;
  y = 15;

  std::cout << "Hello, World!\n";
  std::cout << "x = " << x << " y = " << y << "\n";
  double res = lal::SimpleSum(x, y);
  std::cout << "x = " << x << " y = " << y << " x + y = " << res <<"\n";
}
