#ifndef ML_HPP
#define ML_HPP

#include <iostream>
#include <LinearAlgebra.hpp>

namespace sml {

class LinearRegression {
private:
  sla::Matrix<float> beta = sla::zeros(1, 1);
public:
  void fit(sla::Matrix<float> X, sla::Matrix<float> y);
  sla::Matrix<float> getBeta();
  sla::Matrix<float> predict(sla::Matrix<float> X);
public:
  LinearRegression(); //Constructor
};
}

#endif // ML_HPP
