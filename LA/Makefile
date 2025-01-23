# Nome do compilador
CXX = g++

# Flags de compilação
CXXFLAGS = -Wall -std=c++11

# Nome do executável final
#TARGET_EXEC = main
TARGET_LIB = liblinalg.a

# Arquivos fonte
SRCS_CPP = LinearAlgebra.cpp matrix.cpp

# Arquivos objeto
OBJS_CPP = $(SRCS_CPP:.cpp=.o)
all: $(TARGET_LIB) #$(TARGET_EXEC)

# Criar a biblioteca estática a partir de objetos de C++ e Fortran
$(TARGET_LIB): $(OBJS_CPP)
	ar rcs $@ $(OBJS_CPP)

# Regra para compilar o executável e linkar com a biblioteca
#$(TARGET_EXEC): $(TARGET_LIB)
#	$(CXX) $(CXXFLAGS) -o $@ main.o -L. -lmylib -lgfortran

# Regras para compilar arquivos .cpp em .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regras para compilar arquivos .f90 em .o

# Limpeza dos arquivos gerados
clean:
	rm -f $(OBJS_CPP) $(OBJS_F90) $(TARGET_LIB) $(TARGET_EXEC)
