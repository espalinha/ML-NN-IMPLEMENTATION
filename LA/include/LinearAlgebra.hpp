#ifndef LINEARALGEBRA_HPP
#define LINEARALGEBRA_HPP

#include <LinearAlgebra.hpp>
#include <iostream>
#include <pthread.h>
#include <vector>

//Linear Algebra Lib
namespace sla{
  template<class T>
  struct Vec{
    T* arr;
    int size;
  };

 template<class T>
  class Matrix
  { 
    public:
      void printMatrix();
      T* linearMatrix();
      Matrix<T> transpose();
      Matrix<T> copy();
      Matrix<double> copy_();
      Matrix<double> I();
      Matrix<T> getCol(int col, int row_start); //its similar: arr[row_start:col] 
      Matrix<T> getRow(int row, int col_start); //its similar: arr[row:col_start] 
      Matrix<T> underRow(int row);
      void exchRow(int row1, int row2);
    public:
      Matrix(T** matrix, int row, int cols); //Constructor
      Matrix(T* matrix, int row, int cols);
      Matrix(std::vector<std::vector<T>> vec); //Constructor
      T* operator[](int row); //represent matrix
      T& operator()(int row); //Represent vector
      Matrix<T> operator+(Matrix<T> B); //Implementar a some duas matrizes
      Matrix<T> operator-(Matrix<T> B); //Implementar a subtração
      Matrix<T> operator*(Matrix<T> A); //Implementar a multiplicação, como ela é;                 
      Matrix<T> operator*(T b); //Implementar a multiplicação, como ela é;     

      Matrix<T> operator/(double fact); //Implementar a some duas matrizes
       
      int rows, cols;
    private:      
      T* arrayMatrix;
    };

    
  typedef struct LU{
    Matrix<double> L;
    Matrix<double> U;
  } LU;
  
  typedef struct QR {
    Matrix<double> Q;
    Matrix<double> R;
  } QR;

  typedef struct PLU {
    Matrix<double> P;
    Matrix<double> L;
    Matrix<double> U;
  } PLU;

  template<class T>
  Matrix<T> solve(Matrix<T> A, Matrix<T> b);

  template<class T>
  Matrix<T> forward_substitution(Matrix<T> L, Matrix<T> b);

  template<class T>
  Matrix<T> back_substitution(Matrix<T> U, Matrix<T> y);

  template<class T>
  Matrix<T> cholesky(Matrix<T> A);

  template<class T>
  PLU luDecomp(Matrix<T> m);

  Matrix<double> I(int n);

  template<class T>
  Matrix<T> getCol(int col, int row_start, T* arr, int row_size); //its similar: arr[row_start:col] 
//
  template<class T>
  Matrix<T> getRow(int row, int col_start, T* arr, int col_size); //its similar: arr[row:col_start] 
//
  template<class T>
  Matrix<T> underRow(int row, T* arr, int row_size, int col_size);

  template<class T>
  bool isclose(T x, double y);

  template<class T>
  Matrix<double> zeros(Matrix<T> x);

  Matrix<double> zeros(int row, int col);

  template<class T>
  Matrix<double> inv(Matrix<T> A);

  template<class T>
  double det(Matrix<T> A);

  template<class T>
  QR qr(Matrix<T> A);

  template<class T>
  Matrix<double> eig(Matrix<T> A, double tolerance=1e-12, int maxinter=1000);

  template<class T>
  double normVec(Matrix<T> A);

}

#endif // LINEARALGEBRA_HPP
