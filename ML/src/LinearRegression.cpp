#include "ml.hpp"
#include <LinearAlgebra.hpp>

namespace sml{
LinearRegression::LinearRegression() {
  this->beta = sla::zeros(1,1);
}

void LinearRegression::fit(sla::Matrix<float> X, sla::Matrix<float> y) {
  X = sla::concatenate(sla::ones(X.rows, 1), X);
  auto beta_ = sla::zeros(1, X.cols); 
  beta_ = sla::inv(X.transpose()*X);   
  beta_ = beta_*X.transpose();   
  beta_ = beta_*y;   
  this->beta = beta_; 
}

sla::Matrix<float> LinearRegression::getBeta() {
  return this->beta;
}

sla::Matrix<float> LinearRegression::predict(sla::Matrix<float> X) {
  return sla::ones(X.rows, 1)*this->beta(0) + X*this->beta(1);
}

}
