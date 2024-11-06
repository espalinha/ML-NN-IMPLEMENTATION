#ifndef LINEARALGEBRA_HPP
#define LINEARALGEBRA_HPP

#include <iostream>
#include <pthread.h>

extern "C" {
    void simpleSumFortran(double* x, double* y);
}


//Linear Algebra Lib
namespace lal{

  double SimpleSum(double x, double y);
  class Matrix
  {
    public:
      
    public:
      //Matrix(double* x, double* y); //Constructor
      
    private:
      //int ratio;
  };
}


#endif // LINEARALGEBRA_HPP