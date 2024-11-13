! example.f90
program main
  use :: unix
  implicit none
  type :: pthread_struct
    integer :: init
    integer :: final
  end type pthread_struct
  integer, parameter :: NTHREADS = 15
  integer, dimension(:), allocatable :: arr1, arr2, ans
  integer, parameter :: size = 100
  integer           :: i, rc
  integer :: tam, r !Quantidade de elementos por threads
  integer, target   :: routines(NTHREADS) = [ (i, i = 1, NTHREADS) ]
  type(pthread_struct), dimension(:), allocatable :: structs 
  type(c_pthread_t) :: threads(NTHREADS)
  type(c_ptr)       :: ptr

  !--------------- Estrutura para a pthread -------------
  
  !--------------- Estrutura para a pthread -------------

  allocate(arr1(size))
  allocate(arr2(size))
  allocate(ans(size))
  allocate(structs(NTHREADS))

  do i = 1, size
    arr1(i) = i
    arr2(i) = 10*i
  end do

  
!  do i = 1, size
 !   print *, arr1(i)
  !  print *, arr2(i)
  !end do

  tam = size/NTHREADS
  r = mod(size, NTHREADS)

  ! Create threads.
  do i = 1, NTHREADS
    structs(i)%init = tam*(i - 1)
    structs(i)%final = tam*(i)

    !print *, 'Thread ', i, ' init: ', structs(i)%init, ' final:', structs(i)%final
    rc = c_pthread_create(threads(i), c_null_ptr, c_funloc(sum_vec), c_loc(routines(i)))
  end do                                                                !Qual a thread que estamos usando

  do i = 1, NTHREADS
    rc = c_pthread_join(threads(i), ptr)
  end do
  !Com a quantidade restante
  if (r /= 0) then
    do i = 1, r
      structs(i)%init = tam*NTHREADS + i - 1
      structs(i)%final = tam*NTHREADS + i
      !print *, ' init: ', structs(i)%init, ' final:', structs(i)%final
      rc = c_pthread_create(threads(i), c_null_ptr, c_funloc(sum_vec), c_loc(routines(i)))
    end do
  end if
  print '("Waiting for threads to finish ...")'

  ! Join threads.
  do i = 1, r
      rc = c_pthread_join(threads(i), ptr)
  end do
  

  
  do i = 1, size
    write(*,'(1x,i0)',advance='no') arr1(i)  
  end do
  print *, ""
  do i = 1, size
    write(*,'(1x,i0)',advance='no') arr2(i)  
  end do
  print *, ""
  do i = 1, size
    write(*,'(1x,i0)',advance='no') ans(i)  
  end do
  print *, ""
contains
  recursive subroutine sum_vec(arg) bind(c)
      !! Runs inside a thread and prints out current thread id and loop
      !! iteration.
      type(c_ptr), intent(in), value :: arg !! Client data as C pointer.

      integer, pointer :: n ! Fortran pointer to client data.
      integer          :: i, rc
      integer :: init, final
      if (.not. c_associated(arg)) return ! Check whether argument has been passed.
      call c_f_pointer(arg, n)            ! Convert C pointer to Fortran pointer.
      init = structs(n)%init
      final = structs(n)%final
      do i = init, final
          !print '("--- Thread #", i0, " - Loop iteration ", i0)', n, i
          !rc = c_usleep(10**6) ! Sleep 1 sec.
        ans(i) = arr1(i) + arr2(i)          
      end do
  end subroutine sum_vec
end program main