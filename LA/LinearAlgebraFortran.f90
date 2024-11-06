module LinearAlgebra
  use, intrinsic :: iso_c_binding, only: c_double
  implicit none

contains

  subroutine simpleSumFortran(x, y) bind(C, name="simpleSumFortran")
    real(c_double), intent(in) :: x
    real(c_double), intent(out) :: y
    y = x + y
  end subroutine simpleSumFortran

end module LinearAlgebra