#ifndef LINEARALGEBRA_HPP
#define LINEARALGEBRA_HPP

#include <LinearAlgebra.hpp>
#include <cstddef>
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
    Matrix<float> copy_();
    Matrix<float> I();
    Matrix<T> getCol(int col, int row_start); //its similar: arr[row_start:col] 
    Matrix<T> getRow(int row, int col_start); //its similar: arr[row:col_start] 
    Matrix<T> underRow(int row);
    void exchRow(int row1, int row2);
    std::vector<T> stdvector();
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

    Matrix<T> operator/(float fact); //Implementar a some duas matrizes
     
    int rows, cols;
  private:      
    T* arrayMatrix;
  };

  
typedef struct LU{
  Matrix<float> L;
  Matrix<float> U;
} LU;

typedef struct QR {
  Matrix<float> Q;
  Matrix<float> R;
} QR;

typedef struct PLU {
  Matrix<float> P;
  Matrix<float> L;
  Matrix<float> U;
} PLU;

typedef struct EIG {
  Matrix<float> eigVals;
  Matrix<float> eigVecs;
} EIG;

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

Matrix<float> I(int n);

template<class T>
Matrix<T> getCol(int col, int row_start, T* arr, int row_size); //its similar: arr[row_start:col] 
//
template<class T>
Matrix<T> getRow(int row, int col_start, T* arr, int col_size); //its similar: arr[row:col_start] 
//
template<class T>
Matrix<T> underRow(int row, T* arr, int row_size, int col_size);

template<class T>
bool isclose(T x, float y);

template<class T>
Matrix<float> zeros(Matrix<T> x);

Matrix<float> zeros(int row, int col);

template<class T>
Matrix<float> ones(Matrix<T> x);

Matrix<float> ones(int row, int col);



template<class T>
Matrix<float> inv(Matrix<T> A);

template<class T>
float det(Matrix<T> A);

template<class T>
QR qr(Matrix<T> A);

template<class T>
EIG eig(Matrix<T> A, float tolerance=1e-12, int maxinter=1000);

template<class T>
float normVec(Matrix<T> A);

template<class T>
Matrix<float> concatenate(Matrix<T> A, Matrix<T> B);

Matrix<float> linspace(float a, float b, size_t size=0);

}

#endif // LINEARALGEBRA_HPP
