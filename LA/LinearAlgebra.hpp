#ifndef LINEARALGEBRA_HPP
#define LINEARALGEBRA_HPP

#include <iostream>
#include <pthread.h>
#include <vector>

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
  class Matrix
  { 
    public:
      void printMatrix();
    public:
      Matrix(double** matrix, int row, int cols); //Constructor
      //double** getMatrix(); //To get in the matrix format
      //Matrix(std::vector<std::vector<double>> vec); //Constructor
      double* operator[](int row);
      double* operator+(Matrix A, Matrix B); //Implementar a some duas matrizes
      double* operator-(Matrix A, Matrix B); //Implementar a subtração
      double* operator*(Matrix A, Matrix B); //Implementar a multiplicação, como ela é;
      ~Matrix();
      
    private:      
      double* arrayMatrix;
      int rows, cols;
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
  
  double** ConvertToMatrix(double* linearArray, int size);

  double** MatrixSum(double** A, double** B, int size); //Make it for all types of matrix

  double SimpleSum(double x, double y);

  double innerProduct(double* x, double* y, int size);
}


#endif // LINEARALGEBRA_HPP