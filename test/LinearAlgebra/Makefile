# Nome do compilador
CXX = g++

# Flags de compilação
CXXFLAGS = -Wall -std=c++11 -I../../LA/include

LINALGPATH = ../../LA/

FORTRANLINK = -lgfortran
# Nome do executável
TARGET = latest

# Arquivos fonte
SRCS = latest.cpp

# Arquivos objeto
OBJS = $(SRCS:.cpp=.o)

# Regra padrão para construir o executável
all: $(TARGET)

# Compilar o executável
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -L$(LINALGPATH) -llinalg ${FORTRANLINK}

# Regra para compilar arquivos .cpp em .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Limpeza dos arquivos objeto e do executável
clean:
	rm -f $(OBJS) $(TARGET)
