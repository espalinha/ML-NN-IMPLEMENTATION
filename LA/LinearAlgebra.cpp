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
}

namespace lal{
  void Matrix::printMatrix() {
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        std::cout << arrayMatrix[i * cols + j] << " ";
      }
      std::cout << "\n";
    }
  }

  Matrix::Matrix(double** matrix, int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    arrayMatrix = (double*)malloc(sizeof(double)*rows*cols);
    for(int j = 0; j < rows; j++) {
      for(int i = 0; i < cols; i++) {
        arrayMatrix[i*cols + j] = matrix[i][j];
      }
    }
  }
  /*
  double** Matrix::getMatrix(){
    return lal::ConvertToMatrix(arrayMatrix, rows*cols);
  }
  */

  double* Matrix::operator[](int row) {
        if (row < 0 || row >= rows) {
            throw std::out_of_range("Index out of range");
        }
        return &arrayMatrix[row * cols];
    }

  Matrix::~Matrix() {
    free(this->arrayMatrix);
  }
}