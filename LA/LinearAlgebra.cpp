#include <iostream>
#include <pthread.h>
#include <math.h>
#include "LinearAlgebra.hpp"
namespace lal
{ 
  double** ConvertToMatrix(double* linearArray, int size) {
    // Alocar memória para a matriz bidimensional
    double** matrix = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*)malloc(size * sizeof(double));
    }

    // Copiar valores do array unidimensional para a matriz
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = linearArray[i * size + j];
        }
    }
    
    return matrix;
  }

  double** MatrixSum(double** A, double** B, int size) { 
    double* A_linear = (double*)malloc(size * size * sizeof(double));
    double* B_linear = (double*)malloc(size * size * sizeof(double));
    double* res_linear = (double*)malloc(size * size * sizeof(double));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A_linear[i * size + j] = A[i][j];
            B_linear[i * size + j] = B[i][j];
            res_linear[i * size + j] = 0;
        }
    }

    MatrixSumFortran(A_linear, B_linear, res_linear, size);

    free(A_linear);
    free(B_linear);

    double** res = ConvertToMatrix(res_linear, size);


    return res;
  }

  double innerProduct(double* x, double* y, int size){
    double res;
    innerProductFortran(x, y, &res, size);
    return res;
  }

  double SimpleSum(double x, double y) {
    simpleSumFortran(&x, &y);
    return y;
  }
}