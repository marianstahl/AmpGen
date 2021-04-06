#!/bin/bash

. $(conda info --root)/etc/profile.d/conda.sh
conda activate env_${TRAVIS_OS_NAME}

echo -en 'travis_fold:start:script.build\\r'
echo "Building..."
echo "Building under OS: $TRAVIS_OS_NAME"

mkdir -p build
cd build
echo "CMake-ing, CXX = $CXX"
cmake .. -DCMAKE_CXX_COMPILER=clang -DUSE_SIMD="" -DUSE_OPENMP=0
echo "Building ..."
cmake --build . -- -j2 
cd ..
echo "Running test job ..."
./build/bin/Generator options/example_b2kstarll.opt --CompilerWrapper::Verbose --nEvents 10000
