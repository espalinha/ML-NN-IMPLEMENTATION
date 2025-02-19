cmake_minimum_required(VERSION 3.14)
project(ML VERSION 1.0 LANGUAGES CXX)

include(FetchContent)

# Diretórios
set(LINALGDIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/LA/src)
set(LINALGDIR_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/LA/include)

set(MLDIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ML/src)
set(MLDIR_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ML/include)


# Alvo da biblioteca
add_library(linalg SHARED ${LINALGDIR_SRC}/matrix.cpp)

# Inclua o diretório de cabeçalhos
target_include_directories(linalg PUBLIC ${LINALGDIR_INCLUDE})

add_library(mllib SHARED ${MLDIR_SRC}/LinearRegression.cpp ${MLDIR_SRC}/LogisticRegression.cpp)

# Inclua o diretório de cabeçalhos
target_include_directories(mllib PUBLIC ${MLDIR_INCLUDE})


# Instalação
install(TARGETS linalg DESTINATION lib)
install(TARGETS mllib DESTINATION lib)

install(DIRECTORY ${LINALGDIR_INCLUDE}/ DESTINATION include)
install(DIRECTORY ${MLDIR_INCLUDE}/ DESTINATION include)


