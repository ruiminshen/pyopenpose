# Python components of the [OpenPose](https://arxiv.org/abs/1611.08050)

This project provides Python modules for the [OpenPose in PyTorch](https://github.com/ruiminshen/openpose-pytorch) project. Two types of components are implemented in C++ to accelerate the running efficiency and engineering deployment:

* Preprocessor: Converting the original keypoint coordinate label into two types (the Gaussian heatmap for keypoints and the heatmap for part affinity fields) of feature labels.
* Postprocessor: Analyzing the feature maps obtained by the deep neural network (DNN) to get the keypoint coordinates of multiple objects (usually person).

If you are benefited from this project, a donation will be appreciated (via [PayPal](https://www.paypal.me/minimumshen), [微信支付](donate_mm.jpg) or [支付宝](donate_alipay.jpg)).

## Dependent tools and libraries

* Linux is recommended, and Windows is not tested.
* Compiler supports C++11 or later (such as [GCC](https://gcc.gnu.org/) 4.8 or later).
* [Python](https://www.python.org/) development library, version 3 is recommended.
* [CMake](https://cmake.org/).
* [pybind11](https://github.com/pybind/pybind11).
* [Boost](https://www.boost.org/).
* [Eigen3](http://eigen.tuxfamily.org).
* [OpenCV3](https://opencv.org/).
* [cnpy](https://github.com/rogersce/cnpy) (optional).

## Install

* Install the depend tools and libraries.
* Download and extract the source code (e.g., `~/code/pyopenpose`).
* Create the build directory (e.g., `~/code/pyopenpose-build`) and generate CMake cache (`mkdir ~/code/pyopenpose-build && cd ~/code/pyopenpose-build && cmake ~/code/pyopenpose`)
* Check if the following variables in `~/code/pyopenpose-build/CMakeCache.txt` are correct:
    - `PYMODULE_ROOT`: should be the user location for Python modules (e.g., `~/.local/lib/python3.5/site-packages`)
    - `PYTHON_EXECUTABLE` and `PYTHON_LIBRARY`: should be the interpreter and library of the same version of Python. Version 3 is recommended (e.g., `/usr/bin/python3` and `/usr/lib/x86_64-linux-gnu/libpython3.5m.so`).
    - C++11 is enabled (the `CMAKE_CXX_FLAGS` is set to `-std=c++11 -fPIC` by default).
    - `DEBUG_SHOW` can be used to debug the program in test projects (in the `test` folder of the source directory).
* Compile the source code (`cd ~/code/pyopenpose-build && make`).

If everything goes well, a dynamic library (e.g., `~/.local/lib/python3.5/site-packages/pyopenpose.cpython-35m-x86_64-linux-gnu.so`) which contains the Python modules will be appeared and can be used in Python codes.