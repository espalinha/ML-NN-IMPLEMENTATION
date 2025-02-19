#include "LinearAlgebra.hpp"
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <vector>
#include <math.h>

namespace sla
{

template <class T>
void Matrix<T>::printMatrix()
{
  for (int i = 0; i < this->rows; i++)
  {
    for (int j = 0; j < this->cols; j++)
    {
      std::cout << this->arrayMatrix[i * cols + j] << " ";
    }
    std::cout << "\n";
  }
}

template <class T>
Matrix<T>::Matrix(T **matrix, int rows, int cols)
{
  this->rows = rows;
  this->cols = cols;
  this->arrayMatrix = new T[rows * cols]();
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      arrayMatrix[i * cols + j] = matrix[i][j];
    }
  }
}

template <class T>
Matrix<T>::Matrix(T *matrix, int rows, int cols)
{
  this->rows = rows;
  this->cols = cols;
  this->arrayMatrix = new T[rows * cols]();
  for (int i = 0; i < rows * cols; i++)
  {
    arrayMatrix[i] = matrix[i];
  }
}
template <class T>
Matrix<T>::Matrix(std::vector<std::vector<T>> vec)
{
  this->rows = vec.size();
  this->cols = vec.at(0).size();
  this->arrayMatrix = new T[rows * cols]();
  for (int i = 0; i < this->rows; i++)
  {
    for (int j = 0; j < this->cols; j++)
    {
      arrayMatrix[i * this->cols + j] = vec.at(i).at(j);
    }
  }
}
/*
double** Matrix::getMatrix(){
  return lal::ConvertToMatrix(arrayMatrix, rows*cols);
}
*/

template <class T>
T *Matrix<T>::operator[](int row)
{
  if (rows == 1 || cols == 1)
  {
    return &arrayMatrix[row];
  }
  if (row < 0 || row >= this->rows)
  {
    throw std::out_of_range("Index out of range");
  }
  return &arrayMatrix[row * cols];
}

template <class T>
T &Matrix<T>::operator()(int index)
{
  if (rows == 1 || cols == 1)
  { // É um vetor
    if (index < 0 || index >= rows * cols)
    {
      throw std::out_of_range("Index out of range");
    }
    return arrayMatrix[index];
  }
  throw std::logic_error("Use operator[](int row) para matrizes!");
}
template <class T>
T *dot(T *a, T *b, int row, int col, int mid)
{
  T *ans = new T[row * col]();
  if (!ans)
  {
    std::cout << "error dot\n";
  }
  // std::cout << "alloc\n";
  for (int i = 0; i < row * col; i++)
  {
    ans[i] = 0; // Inicializa cada elemento com 0
  }
  for (int i = 0; i < row; i++)
  {
    for (int k = 0; k < mid; k++)
    {
      for (int j = 0; j < col; j++)
      {
        ans[i * col + j] += a[i * mid + k] * b[k * col + j];
      }
    }
  }
  return ans;
}

template <class T>
void Matrix<T>::exchRow(int row1, int row2)
{
  for (int k = 0; k < this->cols; k++)
  {
    T temp = this->arrayMatrix[row1 * this->cols + k];
    this->arrayMatrix[row1 * this->cols + k] = this->arrayMatrix[row2 * this->cols + k];
    this->arrayMatrix[row2 * this->cols + k] = temp;
  }
}

template <class T>
T *Matrix<T>::linearMatrix()
{
  return this->arrayMatrix;
}

template <class T>
T *parallelDot(T *a, T *b, int row, int col, int mid)
{
  return a;
}

template <class T>
Matrix<T> Matrix<T>::operator*(Matrix<T> m)
{
  if (this->cols != m.rows)
  {
    throw std::invalid_argument("The matrices are dimensional incompability");
  }
  T *ans = dot(this->arrayMatrix, m.linearMatrix(), this->rows, m.cols, this->cols);
  // T* ans = parallelDot(this->arrayMatrix, m.linearMatrix(), this->rows, m.cols, this->cols);

  return Matrix<T>(ans, this->rows, m.cols);
}
template <class T>
Matrix<T> Matrix<T>::operator*(T m)
{
  T *ans = new T[this->rows * this->cols]();
  for (int i = 0; i < this->rows; i++)
  {
    for (int j = 0; j < this->cols; j++)
    {
      ans[i * this->cols + j] = this->arrayMatrix[i * cols + j] * m;
    }
  }
  return Matrix<T>(ans, this->rows, this->cols);
}

template <class T>
T *matrixSum(T *a, T *b, int sig, int row, int col)
{
  T *ans = new T[row * col]();
  if (!ans)
  {
    std::cout << "error\n";
  }
  // std::cout << "alloc\n";
  for (int i = 0; i < row * col; i++)
  {
    ans[i] = 0; // Inicializa cada elemento com 0
  }

  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      ans[i * col + j] = a[i * col + j] + sig * b[i * col + j];
    }
  }

  return ans;
}

template <class T>
Matrix<T> Matrix<T>::operator+(Matrix<T> m)
{
  if (this->rows != m.rows && this->cols != m.cols)
  {
    throw std::invalid_argument("The matrices are dimensional incompability");
  }
  T *ans = matrixSum(this->arrayMatrix, m.linearMatrix(), 1, m.rows, m.cols);

  return Matrix<T>(ans, this->rows, m.cols);
}
template <class T>
Matrix<T> Matrix<T>::operator/(double factor)
{
  if (factor == 0)
    throw std::invalid_argument("division by 0");
  Matrix<T> ans = Matrix<T>(this->arrayMatrix, this->rows, this->cols);
  for (int i = 0; i < this->rows; i++)
  {
    for (int j = 0; j < this->cols; j++)
    {
      ans.arrayMatrix[i * this->cols + j] /= factor;
    }
  }
  return ans;
}

template <class T>
Matrix<T> getCol(int col, int row_start, T *arr, int row_size, int col_size)
{
  int tam = (row_size - row_start);
  T *ans = new T[tam];
  for (int i = row_start; i < row_size; i++)
  {
    ans[i - row_start] = arr[i * col_size + col];
  }
  return Matrix<T>(ans, tam, 1);
}
template <class T>
Matrix<T> getRow(int row, int col_start, T *arr, int col_size, int row_size)
{
  int tam = (col_size - col_start);
  T *ans = new T[tam];
  for (int i = col_start; i < col_size; i++)
  {
    ans[i - col_start] = arr[col_size * row + i];
  }
  return Matrix<T>(ans, 1, tam);
}

template <class T>
Matrix<T> Matrix<T>::getCol(int col, int row_start)
{
  return sla::getCol(col, row_start, this->arrayMatrix, this->rows, this->cols);
}
template <class T>
Matrix<T> Matrix<T>::getRow(int row, int col_start)
{
  return sla::getRow(row, col_start, this->arrayMatrix, this->cols, this->rows);
}

template <class T>
Matrix<T> Matrix<T>::transpose()
{
  int row = this->rows;
  int col = this->cols;
  // std::cout << "r " << row << " c " << col << "\n";
  T *ans = new T[col * row]();
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      ans[j * row + i] = this->arrayMatrix[i * col + j];
    }
  }

  return Matrix<T>(ans, col, row);
}

template <class T>
Matrix<T> Matrix<T>::operator-(Matrix<T> m)
{
  if (this->rows != m.rows && this->cols != m.cols)
  {
    throw std::invalid_argument("The matrices are dimensional incompability");
  }
  T *ans = matrixSum(this->arrayMatrix, m.linearMatrix(), -1, m.rows, m.cols);

  return Matrix<T>(ans, this->rows, m.cols);
}

Matrix<double> zeros(int row, int col)
{
  double *ans = new double[row * col]();
  for (int i = 0; i < row; i++)
  {
    for (int k = 0; k < col; k++)
    {
      ans[i * col + k] = 0.;
    }
  }
  return sla::Matrix<double>(ans, row, col);
}
template <class T>
Matrix<double> zeros(Matrix<T> x)
{

  double *ans = new double[x.rows * x.cols];
  for (int i = 0; i < x.rows; i++)
  {
    for (int j = 0; j < x.cols; j++)
    {
      ans[i * x.cols + j] = 0.;
    }
  }
  return Matrix<T>(ans, x.rows, x.cols);
}

template <class T>
Matrix<T> underRow(int row, T *arr, int row_size, int col_size)
{
  int tam = (row_size - row);
  T *ans = new T[tam * col_size]();
  for (int i = row; i < row_size; i++)
  {
    for (int j = 0; j < col_size; j++)
    {
      ans[(i - row) * col_size + j] = arr[i * col_size + j];
    }
  }
  return Matrix<T>(ans, tam, col_size);
}

template <class T>
Matrix<T> Matrix<T>::underRow(int row)
{
  return sla::underRow(row, this->arrayMatrix, this->rows, this->cols);
}

template <class T>
Matrix<T> Matrix<T>::copy()
{
  int row = this->rows;
  int col = this->cols;
  T ans[row * col];
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      ans[i * col + j] = this->arrayMatrix[i * col + j];
    }
  }
  return Matrix<T>(ans, row, col);
}
template <class T>
Matrix<double> Matrix<T>::copy_()
{
  double *ans = new double[this->rows * this->cols]();
  for (int i = 0; i < this->rows; i++)
  {
    for (int j = 0; j < this->cols; j++)
    {
      ans[i * this->cols + j] = (double)(this->arrayMatrix[i * this->cols + j]);
    }
  }
  return Matrix<double>(ans, this->rows, this->cols);
}
template <class T>
Matrix<double> Matrix<T>::I()
{
  if (this->rows != this->cols)
    throw std::invalid_argument("Matrix is not square");
  int num = this->rows;
  double ans[num * num];
  for (int i = 0; i < num; i++)
  {
    for (int j = 0; j < num; j++)
    {
      if (j == i)
        ans[i * num + j] = 1.0;
      else
        ans[i * num + j] = 0.0;
    }
  }
  return Matrix<double>(ans, num, num);
}

template <class T>
Matrix<double> inv(Matrix<T> A)
{
  int row = A.rows;
  if (row == 2)
  {
    double det = sla::det(A);
    if (det != 0)
    {
      auto res = sla::zeros(2, 2);
      res[0][0] = A[1][1] / det;
      res[0][1] = -A[0][1] / det;
      res[1][0] = -A[1][0] / det;
      res[1][1] = A[0][0] / det;
      return res;
    }
    else
      throw std::invalid_argument("Matrix is singular");
  }
  Matrix<double> b = sla::I(row);

  Matrix<double> invers = sla::zeros(A.rows, A.cols);
  for (int i = 0; i < row; i++)
  {
    auto b_slice = b.getCol(i, 0);
    auto a_inv = sla::solve(A, b_slice);
    for (int j = 0; j < row; j++)
    {
      invers[j][i] = a_inv(j);
    }
  }
  return invers;
}

template <class T>
Matrix<T> solve(Matrix<T> A, Matrix<T> b)
{
  auto x = luDecomp(A);
  auto b_perm = x.P * b;
  Matrix<T> y = sla::forward_substitution(x.L, b_perm);
  return sla::back_substitution(x.U, y);
}

template <class T>
Matrix<T> forward_substitution(Matrix<T> L, Matrix<T> b)
{
  int row = L.rows;
  Matrix<T> y = sla::zeros(b);
  y(0) = b(0) / L[0][0];
  for (int i = 1; i < row; i++)
  {
    T sum = 0;
    for (int k = 0; k < i; k++)
    {
      sum += L[i][k] * y(k);
    }
    y(i) = (b(i) - sum) / L[i][i];
  }
  return y;
}

template <class T>
Matrix<T> back_substitution(Matrix<T> U, Matrix<T> y)
{
  int row = U.rows;
  Matrix<T> x = sla::zeros(y);
  x(row - 1) = y(row - 1) / U[row - 1][row - 1];
  for (int i = row - 2; i > -1; i--)
  {
    double sum = 0;
    for (int k = i; k < row; k++)
    {
      sum += U[i][k] * x(k);
    }
    x(i) = (y(i) - sum) / U[i][i];
  }
  return x;
}

template <class T>
PLU luDecomp(Matrix<T> m)
{
  int row = m.rows;
  auto U = m.copy_();
  auto L = m.I();
  auto P = m.I();

  for (int i = 0; i < row; i++)
  {
    for (int k = i; k < row; k++)
    {
      if (!sla::isclose(U[i][i], 0.))
        break;
      U.exchRow(k, k + 1);
      P.exchRow(k, k + 1);
    }

    auto factor = U.getCol(i, i + 1) / U[i][i];
    for (int k = i + 1; k < row; k++)
    {
      L[k][i] = factor(k - (i + 1));
    }
    factor = factor * U.getRow(i, 0);
    for (int k = i + 1; k < row; k++)
    {
      for (int p = 0; p < row; p++)
      {
        U[k][p] = U[k][p] - factor[k - (i + 1)][p];
      }
    }
  }
  return (PLU){P, L, U};
}
template <class T>
Matrix<T> cholesky(Matrix<T> A)
{
  int row = A.rows;
  Matrix<T> L = sla::zeros(A);
  for (int i = 0; i < row; i++)
  {
    double sum = 0;
    Matrix<T> temp = A.getRow(i, 0);
    for (int k = 0; k < row; k++)
    {
      sum = temp(k) * temp(k);
    }
    L[i][i] = std::sqrt(A[i][i] - sum);
    Matrix<T> ur = L.underRow(i + 1);
    Matrix<T> allcol = L.getCol(i, 0);
    ur = ur * allcol;
    Matrix<T> Aur = A.underRow(i + 1);
    Matrix<T> Acol = Aur.getCol(i, 0);
    Acol = Acol - ur;
    Acol = Acol / L[i][i];
    for (int k = i + 1; k < row; k++)
    {
      L[k][i] = A[k][i];
    }
  }
  return L;
}
Matrix<double> I(int num)
{
  double ans[num * num];
  for (int i = 0; i < num; i++)
  {
    for (int j = 0; j < num; j++)
    {
      if (j == i)
        ans[i * num + j] = 1.0;
      else
        ans[i * num + j] = 0.0;
    }
  }
  return Matrix<double>(ans, num, num);
}

template <class T>
bool isclose(T x, double y)
{
  double val = std::abs(x - y);
  if (val < 1e-8)
    return true;
  return false;
}

template <class T>
PLU luDecomp_for_det(Matrix<T> m)
{
  int row = m.rows;
  auto U = m.copy_();
  auto L = m.I();
  auto P = m.I();
  int number_perm = 0;

  for (int i = 0; i < row; i++)
  {
    for (int k = i; k < row; k++)
    {
      if (!sla::isclose(U[i][i], 0.))
        break;
      U.exchRow(k, k + 1);
      P.exchRow(k, k + 1);
      number_perm++;
    }

    auto factor = U.getCol(i, i + 1) / U[i][i];
    for (int k = i + 1; k < row; k++)
    {
      L[k][i] = factor(k - (i + 1));
    }
    factor = factor * U.getRow(i, 0);
    for (int k = i + 1; k < row; k++)
    {
      for (int p = 0; p < row; p++)
      {
        U[k][p] = U[k][p] - factor[k - (i + 1)][p];
      }
    }
  }
  auto P_ = sla::zeros(1, 1);
  P_(0) = number_perm;
  return (PLU){P_, L, U};
}
template <class T>
double det(Matrix<T> A)
{
  int row = A.rows;
  if (row == 2)
  {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
  }
  PLU lu = luDecomp_for_det(A);
  double det_res = 1;
  for (int i = 0; i < row; i++)
  {
    det_res *= lu.U[i][i];
  }
  if ((int)lu.P(0) % 2 == 0)
    return det_res;
  else
    return -det_res;
}
template<class T>
QR qr_decomp(Matrix<T> A) {
  int row = A.rows;
  int col = A.cols;
  Matrix<double> Q = sla::zeros(row, row);
  Matrix<double> u = sla::zeros(row, row);
  Matrix<double> R = sla::zeros(row, col);


  for(int p = 0; p < row; p++) {
    u[p][0] = A[p][0];
  }
  double norm = sla::normVec(u.getCol(0, 0));
  for(int p = 0; p < row; p++) {
    Q[p][0] = u[p][0]/norm;
  }
  for(int i = 1; i < row; i++) {
    for(int p = 0; p < row; p++) {
      u[p][i] = A[p][i];
    }
    for(int j = 0; j < row; j++) {
      double scalar = 0;
      for(int p = 0; p < row; p++) {
        scalar += A[p][i]*Q[p][j];
      }    
      for(int p = 0; p < row; p++) {
        u[p][i] = u[p][i] - scalar*Q[p][j];
      }
   }
    for(int p = 0; p < row; p++) {
      Q[p][i] = u[p][i]/sla::normVec(u.getCol(i, 0));
    }
  }

  for(int i = 0; i < row; i++) {
    for(int j = i; j < col; j++) {
      double scalar = 0;
      for(int p = 0; p < row; p++) {
        scalar += A[p][j]*Q[p][i];
      }    
      R[i][j] = scalar;
    }
  }
   return (QR){Q, R};

  }
  /*
   * Vamos arrumar os sinais, para não termos mais problemas
   * */
template<class T>
QR qr(Matrix<T> A) {
  QR qr__ = sla::qr_decomp(A);
  Matrix<double> D = sla::zeros(A.rows, A.rows);
  for(int i = 0; i < A.rows; i++) {
    if(qr__.Q[i][i] < 0) D[i][i] = -1;
    else if(qr__.Q[i][i] == 0) D[i][i] = 0;
    else D[i][i] = 1;
  }
  qr__.Q = qr__.Q*D;
  qr__.R = D*qr__.R;
  
    return (QR){qr__.Q, qr__.R};
}
template<class T>
Matrix<double> eig(Matrix<T> A, double tolerance, int maxinter) {
  auto A_old = A.copy_();
  auto A_new = A.copy_();
  double diff = INFINITY;
  int i = 0;
  while (diff > tolerance && i < maxinter) {
    A_old = A_new.copy_();
    QR qr = sla::qr_decomp(A_old);
    A_new = qr.R*qr.Q;

    double max = DBL_MIN;
    for(int j = 0; j < A.rows; j++) {
      for(int k = 0; k < A.cols; k++) {
        auto temp = A_new[j][k] - A_old[j][k];
        if(std::abs(temp) > max) max = std::abs(temp);
      }
    }
    diff = max;
    i++;
  }

  Matrix<double> eigs = sla::zeros(1, A.cols);
  for(int i = 0; i < A.rows; i++) {
    eigs(i) = A_new[i][i];
  }
  return eigs;

  
}

template<class T>
double normVec(Matrix<T> A) {
 double norm = 0;
  for(int i = 0; i < A.cols*A.rows; i++) {
    norm += A(i)*A(i);
  }
  
  return std::sqrt(norm);
}



}

template class sla::Matrix<int>;
template class sla::Matrix<double>;
template class sla::Matrix<float>;
template class sla::Matrix<long>;
template class sla::Matrix<short>;
template class sla::Matrix<size_t>;
template class sla::Matrix<uint8_t>;
template class sla::PLU sla::luDecomp<double>(sla::Matrix<double>);
template class sla::Matrix<double> sla::solve<double>(sla::Matrix<double>, sla::Matrix<double>);
template class sla::Matrix<double> sla::cholesky<double>(sla::Matrix<double>);
template <class T>
bool sla::isclose(T, double);
template <class T>
sla::Matrix<double> sla::zeros(Matrix<T>);
template <class T>
sla::Matrix<double> sla::inv(Matrix<T>);
template class sla::Matrix<double> sla::inv<double>(sla::Matrix<double>);
template double sla::det<double>(Matrix<double>);
template double sla::normVec<double>(Matrix<double>);
template sla::QR sla::qr<double>(Matrix<double>);
template sla::Matrix<double> sla::eig<double>(Matrix<double>, double, int);




/*
template sla::LU<double> sla::luDecomp<double>(sla::Matrix<double>);
template sla::LU<double> sla::luDecomp<double>(sla::Matrix<int>);
template sla::LU<double> sla::luDecomp<double>(sla::Matrix<float>);
template sla::LU<double> sla::luDecomp<double>(sla::Matrix<long>);
template sla::LU<double> sla::luDecomp<double>(sla::Matrix<short>);
template sla::LU<double> sla::luDecomp<double>(sla::Matrix<size_t>);
*/
