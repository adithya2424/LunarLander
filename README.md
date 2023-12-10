LunarLander:

A simulation environment for lunar lander using SFML, Box-2D in C++

Experiment.cpp File contains the basic structure for visualzing the lunar lander environment.

Experiment2.cpp File contains the structure + physics simulating the free fall.

Experiment3.cpp File contains the structure + physics + collision errors simulating the free fall

Experiment4.cpp File cotains all of the above structure plus includes ground work for integrating RL algorithms

Running main.cpp

This repository contains a shell script and C++ source file to compile the `main.cpp` file using g++ and required libraries.

Prerequisites

Ensure you have the following installed:

g++ compiler: To compile the C++ source code.
SFML: Simple and Fast Multimedia Library.
Box2D: A 2D physics engine.
LibTorch: PyTorch's C++ library.

Instructions

Follow these steps to compile and run the C++ code using the provided shell script:

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_repository.git
   ```

2. Navigate to the directory:

   ```bash
   cd your_repository
   ```

3. Make the shell script executable:

   ```bash
   chmod +x your_script.sh
   ```

4. Run the shell script:

   ```bash
   ./your_script.sh
   ```

Script Details

The shell script `compile_torch.sh` includes the compilation commands using `g++` along with required library paths and flags to compile `main.cpp` and create the executable `output`.

```bash
#! /bin/sh
g++ -g -std=c++17 main.cpp -o main \
    -I include \
    -I/Lunar_Lander_v2/libtorch/include \
    -I/Lunar_Lander_v2/libtorch/include/torch/csrc/api/include \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -L lib \
    -L/Lunar_Lander_v2/libtorch/lib/ \
    -lsfml-system -lsfml-window -lsfml-graphics -lsfml-audio -lsfml-network -lbox2d -ltorch -lc10 -ltorch_cpu \
    -Wl,-rpath ./lib \
    -o output
```

Please ensure that the paths mentioned in the script (`/Lunar_Lander_v2/libtorch/include`, `/Lunar_Lander_v2/libtorch/lib/`, etc.) match the actual paths on your system.

