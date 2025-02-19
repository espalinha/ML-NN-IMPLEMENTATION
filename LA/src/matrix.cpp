#include "LinearAlgebra.hpp"
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
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
std::vector<T> Matrix<T>::stdvector() {
  return std::vector<T>(this->arrayMatrix, this->arrayMatrix + this->rows*this->cols);
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
float** Matrix::getMatrix(){
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
Matrix<T> Matrix<T>::operator/(float factor)
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

Matrix<float> zeros(int row, int col)
{
  float *ans = new float[row * col]();
  for (int i = 0; i < row; i++)
  {
    for (int k = 0; k < col; k++)
    {
      ans[i * col + k] = 0.;
    }
  }
  return sla::Matrix<float>(ans, row, col);
}
template <class T>
Matrix<float> zeros(Matrix<T> x)
{

  float *ans = new float[x.rows * x.cols];
  for (int i = 0; i < x.rows; i++)
  {
    for (int j = 0; j < x.cols; j++)
    {
      ans[i * x.cols + j] = 0.;
    }
  }
  return Matrix<T>(ans, x.rows, x.cols);
}

Matrix<float> ones(int row, int col)
{
  float *ans = new float[row * col]();
  for (int i = 0; i < row; i++)
  {
    for (int k = 0; k < col; k++)
    {
      ans[i * col + k] = 1.;
    }
  }
  return sla::Matrix<float>(ans, row, col);
}
template <class T>
Matrix<float> ones(Matrix<T> x)
{

  float *ans = new float[x.rows * x.cols];
  for (int i = 0; i < x.rows; i++)
  {
    for (int j = 0; j < x.cols; j++)
    {
      ans[i * x.cols + j] = 1.;
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
Matrix<float> Matrix<T>::copy_()
{
  float *ans = new float[this->rows * this->cols]();
  for (int i = 0; i < this->rows; i++)
  {
    for (int j = 0; j < this->cols; j++)
    {
      ans[i * this->cols + j] = (float)(this->arrayMatrix[i * this->cols + j]);
    }
  }
  return Matrix<float>(ans, this->rows, this->cols);
}
template <class T>
Matrix<float> Matrix<T>::I()
{
  if (this->rows != this->cols)
    throw std::invalid_argument("Matrix is not square");
  int num = this->rows;
  float ans[num * num];
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
  return Matrix<float>(ans, num, num);
}

template <class T>
Matrix<float> inv(Matrix<T> A)
{
  int row = A.rows;
  if (row == 2)
  {
    float det = sla::det(A);
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
  Matrix<float> b = sla::I(row);

  Matrix<float> invers = sla::zeros(A.rows, A.cols);
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
  /*
   * Solve Ax=b
   * */
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
    float sum = 0;
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
    float sum = 0;
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
Matrix<float> I(int num)
{
  float ans[num * num];
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
  return Matrix<float>(ans, num, num);
}

template <class T>
bool isclose(T x, float y)
{
  float val = std::abs(x - y);
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
float det(Matrix<T> A)
{
  int row = A.rows;
  if (row == 2)
  {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
  }
  PLU lu = luDecomp_for_det(A);
  float det_res = 1;
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
  Matrix<float> Q = sla::zeros(row, row);
  Matrix<float> u = sla::zeros(row, row);
  Matrix<float> R = sla::zeros(row, col);


  for(int p = 0; p < row; p++) {
    u[p][0] = A[p][0];
  }
  float norm = sla::normVec(u.getCol(0, 0));
  for(int p = 0; p < row; p++) {
    Q[p][0] = u[p][0]/norm;
  }
  for(int i = 1; i < row; i++) {
    for(int p = 0; p < row; p++) {
      u[p][i] = A[p][i];
    }
    for(int j = 0; j < row; j++) {
      float scalar = 0;
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
      float scalar = 0;
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
  Matrix<float> D = sla::zeros(A.rows, A.rows);
  for(int i = 0; i < A.rows; i++) {
    if(qr__.Q[i][i] < 0) D[i][i] = -1;
    else if(qr__.Q[i][i] == 0) D[i][i] = 0;
    else D[i][i] = 1;
  }
  qr__.Q = qr__.Q*D;
  qr__.R = D*qr__.R;
  
    return (QR){qr__.Q, qr__.R};
}


/*
 * 

"Inverse iteration method to find the eigen vector corresponding to a given eigen value."
function inverseIteration(A, val, maxiter=20, userandomstart=true)
    mu_I = val * eye(A)
    inv_A_minus_muI = inv(A - mu_I)
    n = size(A)[1]
    if userandomstart
        X = randn((n, 1))
    else
        X = ones((n, 1))
    end
    for i = 1:maxiter
        next_X = inv_A_minus_muI * X
        X = next_X / norm(next_X)
    end
    X
end


font: https://perso.crans.org/besson/publis/notebooks/Algorithms_to_compute_eigen_values_and_eigen_vectors_in_Julia.html#:~:text=0.525322;%200.818673%5D)-,QR%20algorithm,on%20the%20Gram%2DSchmidt%20process

 * */

template<class T>
Matrix<float> eigvector(Matrix<T> A, float eigval, float tolerance, int maxinter) {
  auto x = sla::zeros(A.rows, 1);
  auto B = A - A.I()*eigval;

  auto inv = sla::inv(B);
  //inv.printMatrix();
  unsigned seed = 2909;
  std::default_random_engine e(seed);
  
  std::normal_distribution<double> distN(0, 1);

  for(int j = 0; j < A.rows; j++) {
    x(j) = distN(e);
  }
  for (int itter = 0; itter < maxinter; itter++) {
    auto next_x = inv*x;
    //next_x.printMatrix();
    //std::cout << "\n";
    x = next_x/sla::normVec(next_x);
    //x.printMatrix();
  }
  return x;
}


//font: https://www.geeksforgeeks.org/3-way-quicksort-dutch-national-flag/
void swap(Matrix<float> a, int i, int j)
{
    int temp = a(i);
    a(i) = a(j);
    a(j) = temp;
}

void partition(Matrix<float> a, int l, int r, int& i, int& j)
{
    i = l - 1, j = r;
    int p = l - 1, q = r;
    int v = a(r);
 
    while (true) {
        // From left, find the first element greater than
        // or equal to v. This loop will definitely
        // terminate as v is last element
        while (a(++i) < v);
 
        // From right, find the first element smaller than
        // or equal to v
        while (v < a(--j))
            if (j == l)
                break;
 
        // If i and j cross, then we are done
        if (i >= j)
            break;
 
        // Swap, so that smaller goes on left greater goes
        // on right
        swap(a,i,j);
 
        // Move all same left occurrence of pivot to
        // beginning of array and keep count using p
        if (a(i) == v) {
            p++;
            swap(a,p,i);
        }
 
        // Move all same right occurrence of pivot to end of
        // array and keep count using q
        if (a(j) == v) {
            q--;
            swap(a,j,q);
        }
    }
 
    // Move pivot element to its correct index
    swap(a,i,r);
 
    // Move all left same occurrences from beginning
    // to adjacent to arr[i]
    j = i - 1;
    for (int k = l; k < p; k++, j--)
        swap(a,k,j);
 
    // Move all right same occurrences from end
    // to adjacent to arr[i]
    i = i + 1;
    for (int k = r - 1; k > q; k--, i++)
        swap(a,i, k);
}
 
// 3-way partition based quick sort
void quicksort(Matrix<float> a, int l, int r)
{
    
    if (r <= l)
        return;
 
    int i, j;
 
    // Note that i and j are passed as reference
    partition(a, l, r, i, j);
 
    // Recur
    quicksort(a, l, j);
    quicksort(a, i, r);
}

void inverse_vec(Matrix<float> A) {
  for(int i = 0; i < A.cols/2; i++) {
    swap(A, i, A.cols - i - 1);
  }
}

template<class T>
EIG eig(Matrix<T> A, float tolerance, int maxinter) {
  auto A_old = A.copy_();
  auto A_new = A.copy_();
  float diff = INFINITY;
  int i = 0;
  while (diff > tolerance && i < maxinter) {
    A_old = A_new.copy_();
    QR qr = sla::qr_decomp(A_old);
    A_new = qr.R*qr.Q;

    float max = DBL_MIN;
    for(int j = 0; j < A.rows; j++) {
      for(int k = 0; k < A.cols; k++) {
        auto temp = A_new[j][k] - A_old[j][k];
        if(std::abs(temp) > max) max = std::abs(temp);
      }
    }
    diff = max;
    i++;
  }

  Matrix<float> eigs = sla::zeros(1, A.cols);
  for(int i = 0; i < A.rows; i++) {
    eigs(i) = A_new[i][i];
  }
  
  sla::quicksort(eigs, 0, eigs.cols-1);
  sla::inverse_vec(eigs);
  Matrix<float> eigvecs = sla::zeros(A.rows, A.cols);
  for(int i = 0; i < A.cols; i++) {
    auto x = sla::eigvector(A, eigs(i), tolerance, maxinter);
    for(int j = 0; j < A.cols;j++) {
      eigvecs[i][j] = x(j);
    }
  }


  return (EIG) {eigs, eigvecs.transpose()};

  
}

template<class T>
float normVec(Matrix<T> A) {
 float norm = 0;
  for(int i = 0; i < A.cols*A.rows; i++) {
    norm += A(i)*A(i);
  }
  
  return std::sqrt(norm);
}
template<class T>
Matrix<float> concatenate(Matrix<T> A, Matrix<T> B) {
  if(A.rows != B.rows && A.cols != B.cols) {
    throw std::invalid_argument("A and B needs to have, at least, one equal dimension");
  }
  
  Matrix<float> new_matrix = sla::zeros(1, 1);

  if(A.rows == B.rows) {
    new_matrix = sla::zeros(A.rows, (A.cols + B.cols));
    int col = 0;
    for(int i = 0; i < A.rows; i++) {
      for(int j = 0; j < A.cols; j++) {
        new_matrix[i][j] = A[i][j];
      }
    }
    for(int i = 0; i < A.rows;i++) {
      for(int j = A.cols; j < A.cols + B.cols; j++) {
        new_matrix[i][j] = B[i][j - A.cols];
      }
    }
  }

  else {
    new_matrix = sla::zeros(A.rows + B.rows, A.cols);
    int col = 0;
    for(int i = 0; i < A.rows; i++) {
      for(int j = 0; j < A.cols; j++) {
        new_matrix[i][j] = A[i][j];
      }
    }
    for(int i = A.rows; i < A.rows + B.rows;i++) {
      for(int j = 0; j < B.cols; j++) {
        new_matrix[i][j] = B[i - A.rows][j];
      }
    }
  }

  return new_matrix;
}

Matrix<float> linspace(float a, float b, size_t size) {
  size_t size__ = 0;
  if( size == 0 ) {
    size__ = b - a;
  }
  else if (size > 0) {
    size__ = size;
  }
  else {
    throw std::invalid_argument("size must be greater zero");
  }

  float pass = (b - a)/size__;
  Matrix<float> result = sla::zeros(size__, 1);
  for(int i = 0; i < size__; i++) {
    result(i) = a + pass*i;
  }
  return result;
}



}


template class sla::Matrix<int>;
template class sla::Matrix<float>;
template class sla::Matrix<long>;
template class sla::Matrix<short>;
template class sla::Matrix<size_t>;
template class sla::Matrix<uint8_t>;
template class sla::PLU sla::luDecomp<float>(sla::Matrix<float>);
template class sla::Matrix<float> sla::solve<float>(sla::Matrix<float>, sla::Matrix<float>);
template class sla::Matrix<float> sla::cholesky<float>(sla::Matrix<float>);
template <class T>
bool sla::isclose(T, float);
template <class T>
sla::Matrix<float> sla::zeros(Matrix<T>);
template <class T>
sla::Matrix<float> sla::ones(Matrix<T>);

template <class T>
sla::Matrix<float> sla::inv(Matrix<T>);
template class sla::Matrix<float> sla::inv<float>(sla::Matrix<float>);
template float sla::det<float>(Matrix<float>);
template float sla::normVec<float>(Matrix<float>);
template sla::QR sla::qr<float>(Matrix<float>);
template sla::EIG sla::eig<float>(Matrix<float>, float, int);
template sla::Matrix<float> sla::concatenate<float>(Matrix<float>, Matrix<float>);






/*
template sla::LU<float> sla::luDecomp<float>(sla::Matrix<float>);
template sla::LU<float> sla::luDecomp<float>(sla::Matrix<int>);
template sla::LU<float> sla::luDecomp<float>(sla::Matrix<float>);
template sla::LU<float> sla::luDecomp<float>(sla::Matrix<long>);
template sla::LU<float> sla::luDecomp<float>(sla::Matrix<short>);
template sla::LU<float> sla::luDecomp<float>(sla::Matrix<size_t>);
*/
