#include "../../LA/LinearAlgebra.hpp"
#include <iostream>


#define SIZE 10
int main() {
  double x, y;
  x = 10;
  y = 15;
  std::cout << "Hello, World!\n";
  std::cout << "x = " << x << " y = " << y << "\n";
  double res = lal::SimpleSum(x, y);
  std::cout << "x = " << x << " y = " << y << " x + y = " << res <<"\n";

  std::cout << "Inner Product Test\n";
  
  double* x_ = (double*)malloc(sizeof(double)*SIZE);
  double* y_ = (double*)malloc(sizeof(double)*SIZE);

  for(int i = 0; i < SIZE; i++) {
    x_[i] = i + 1;
    y_[SIZE - i -1] = i + 1;
  }
/*
  for(int i = 0; i < SIZE; i++) {
    std::cout << "x = " << x_[i] << " y = " << y_[i] << "\n";
  }
*/

  double** x_m = (double**)malloc(sizeof(double*)*SIZE);
  double** y_m = (double**)malloc(sizeof(double*)*SIZE);


  for(int i = 0; i < SIZE; i++) {
    x_m[i] = (double*)malloc(sizeof(double)*SIZE);
    y_m[i] = (double*)malloc(sizeof(double)*SIZE);
  }

  for(int i = 0; i < SIZE; i++) {
    for(int j = 0; j < SIZE; j++) {
      x_m[i][j] = SIZE - i;
      y_m[i][j] = i + 1;
    }
  }

  for(int i = 0; i < SIZE; i++) {
    for(int j = 0; j < SIZE; j++) {
      std::cout<< x_m[i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  std::cout << "--------------------\n";
  std::cout << "\n";

  for(int i = 0; i < SIZE; i++) {
    for(int j = 0; j < SIZE; j++) {
      std::cout<< y_m[i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  std::cout << "--------------------\n";
  std::cout << "\n";

  double** res_m = lal::MatrixSum(x_m, y_m, SIZE);
  
  for(int i = 0; i < SIZE; i++) {
    for(int j = 0; j < SIZE; j++) {
      std::cout<< res_m[i][j] << " ";
    }
    std::cout << "\n";
  }

  res = lal::innerProduct(x_, y_, SIZE);
  std::cout << "Produto interno: " << res << "\n";

  free(x_);
  free(y_);

  return 0;
}
