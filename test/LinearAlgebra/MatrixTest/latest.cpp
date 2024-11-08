#include "../../../LA/LinearAlgebra.hpp"
#include <iostream>


#define SIZE 3
int main() {
  
  double** matrix = (double**)malloc(sizeof(double*) * SIZE);
  for(int i = 0; i < SIZE; i++) {
    matrix[i] = (double*)malloc(sizeof(double)*SIZE);
  }
  for(int i = 0; i < SIZE; i++) {
    for(int j = 0; j < SIZE; j++) {
      matrix[i][j] = j + 1;
    }  
  }

  for(int i = 0; i < SIZE; i++) {
      for(int j = 0; j < SIZE; j++) {
        std::cout << matrix[i][j] << " ";
      }
      std::cout << "\n";
    }

  lal::Matrix m(matrix, SIZE, SIZE);
  m.printMatrix();

  std::cout << "Valor in m[1][1]: " << m[1][1] << std::endl;
  m[1][1] = 3;
  std::cout << "Valor in m[1][1]: " << m[1][1] << std::endl;
  m.printMatrix();

  for (int i = 0; i < SIZE; i++) {
    delete[] matrix[i];
  }

  return 0;
}
