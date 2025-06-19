#!/bin/bash

if [ "$1" == "train" ]; then
    g++ -fopenmp -O3 -std=c++17 main.cpp -Iinclude -o main && ./main
elif [ "$1" == "traincnn" ]; then
    g++ -fopenmp -O3 -std=c++17 cnn.cpp -Iinclude -o cnn && ./cnn
elif [ "$1" == "test" ]; then
    g++ test.cpp -o test && ./test
elif [ "$1" == "plot" ]; then
    cd ../utils
    python3 plot.py
    cd ../lab6
else
    echo "Uso: $0 [train|test|plot]"
    exit 1
fi