#ifndef LINEARALGEBRA_HPP
#define LINEARALGEBRA_HPP

#include <iostream>
#include <pthread.h>

extern "C" {
    void simpleSumFortran(double* x, double* y);
}

extern "C" {
    //Simple Assume size x size
    void MatrixSumFortran(const double* x, const double* y, double* res, int size);
}

extern "C" {
    void innerProductFortran(double* x, double* y, double* res, int size);
}



//Linear Algebra Lib
namespace lal{
  double** ConvertToMatrix(double* linearArray, int size); //Convert to private

  double** MatrixSum(double** A, double** B, int size); //Make it for all types of matrix

  double SimpleSum(double x, double y);

  double innerProduct(double* x, double* y, int size);
  class Matrix
  {
    public:
      
    public:
      //Matrix(double* x, double* y); //Constructor
      
    private:
      //int ratio;
  };

  /*
    Create the Vector Class
  */

  class Vector
  {
    public:
      
    public:
      //Matrix(double* x, double* y); //Constructor
      
    private:
      //int ratio;
  };
}


#endif // LINEARALGEBRA_HPP