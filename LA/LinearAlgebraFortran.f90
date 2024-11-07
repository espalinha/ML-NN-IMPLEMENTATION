module LinearAlgebra
  use, intrinsic :: iso_c_binding, only: c_double, c_int
  implicit none

contains
  subroutine innerProductFortran(x,  y, res, size) bind(C, name="innerProductFortran")
    integer(c_int), intent(in), value :: size
    real(kind=c_double), dimension(size), intent(in) :: x 
    real(kind=c_double), dimension(size), intent(in)::  y
    real(kind=c_double), intent(out) ::  res

    integer :: i

    do i = 1, size
      res = res + x(i)*y(i)
    end do

  end subroutine innerProductFortran  

  subroutine simpleSumFortran(x, y) bind(C, name="simpleSumFortran")
    real(c_double), intent(in) :: x
    real(c_double), intent(out) :: y
    y = x + y
  end subroutine simpleSumFortran

  !This will be better, only for same saze
  subroutine MatrixSumFortran(x, y, res, size) bind(C, name="MatrixSumFortran")
    integer(c_int), intent(in), value :: size
    real(kind=c_double), dimension(size * size), intent(in) :: x 
    real(kind=c_double), dimension(size * size), intent(in):: y
    real(kind=c_double), dimension(size * size), intent(out) :: res

    integer :: i, j

    do i = 1, size
      do j = 1, size
          !Tipo bloco, igual ao CUDA
          res((j-1)*size + i) = x((j-1)*size + i) + y((j-1)*size + i)
      end do
  end do


  end subroutine MatrixSumFortran

end module LinearAlgebra