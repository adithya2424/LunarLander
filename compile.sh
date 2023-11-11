#! /bin/sh
g++ -std=c++11 experiment2.cpp  -o main -I include -L lib -l sfml-system -l sfml-window -l sfml-graphics -l sfml-audio -l sfml-network -l box2d -Wl,-rpath ./lib