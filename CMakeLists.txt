cmake_minimum_required(VERSION 3.10)
project(ML VERSION 1.0 LANGUAGES CXX)

# Diretórios
set(LINALGDIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/LA/src)
set(LINALGDIR_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/LA/include)


# Alvo da biblioteca
add_library(linalg SHARED ${LINALGDIR_SRC}/matrix.cpp)

# Inclua o diretório de cabeçalhos
target_include_directories(linalg PUBLIC ${LINALGDIR_INCLUDE})

# Instalação
install(TARGETS linalg DESTINATION lib)
install(DIRECTORY ${LINALGDIR_INCLUDE}/ DESTINATION include)

