#include <iostream>
#include <pthread.h>
#include <math.h>
#include "LinearAlgebra.hpp"
namespace lal
{
  double SimpleSum(double x, double y) {
    simpleSumFortran(&x, &y);
    return y;
  }
}