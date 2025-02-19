#!/bin/bash

rm -rf build

mkdir build

cd build && cmake .. && make && sudo make install && cd ..
#installing the matplotplusplus lib

cd ./3party/matplotplusplus/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2" -DMATPLOTPP_BUILD_EXAMPLES=OFF -DMATPLOTPP_BUILD_TESTS=OFF && sudo cmake --build . --parallel 2 --config Release && sudo cmake --install .




