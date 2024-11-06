# Nome do compilador
CXX = g++
FTR = gfortran

# Flags de compilação
CXXFLAGS = -Wall -std=c++11
FTRFLAGS = -fno-underscoring

# Nome do executável final
#TARGET_EXEC = main
TARGET_LIB = liblinalg.a

# Arquivos fonte
SRCS_CPP = LinearAlgebra.cpp
SRCS_F90 = LinearAlgebraFortran.f90

# Arquivos objeto
OBJS_CPP = $(SRCS_CPP:.cpp=.o)
OBJS_F90 = $(SRCS_F90:.f90=.o)

all: $(TARGET_LIB) #$(TARGET_EXEC)

# Criar a biblioteca estática a partir de objetos de C++ e Fortran
$(TARGET_LIB): $(OBJS_CPP) $(OBJS_F90)
	ar rcs $@ $(OBJS_CPP) $(OBJS_F90)

# Regra para compilar o executável e linkar com a biblioteca
#$(TARGET_EXEC): $(TARGET_LIB)
#	$(CXX) $(CXXFLAGS) -o $@ main.o -L. -lmylib -lgfortran

# Regras para compilar arquivos .cpp em .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regras para compilar arquivos .f90 em .o
%.o: %.f90
	$(FTR) $(FTRFLAGS) -c $< -o $@

# Limpeza dos arquivos gerados
clean:
	rm -f $(OBJS_CPP) $(OBJS_F90) $(TARGET_LIB) $(TARGET_EXEC)
