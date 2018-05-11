###
# Copyright (c) 2017, 申瑞珉 (Ruimin Shen)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#	list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#	this list of conditions and the following disclaimer in the documentation
#	and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# add set(CMAKE_CXX_COMPILER_ARG1 "-std=c++11") to CMakeFindEclipseCDT4.cmake if Eclipse CDT is used

cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release MinSizeRel RelWithDebInfo)
endif()

if("${CMAKE_CXX_FLAGS}" STREQUAL "")
	set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC" CACHE STRING "compile flags" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

FIND_PACKAGE(PythonInterp REQUIRED)
if(PYTHONINTERP_FOUND)
	execute_process(COMMAND ${PYTHON_EXECUTABLE} -m site --user-site OUTPUT_VARIABLE PYTHON_USER_SITE_PACKAGE)
	STRING(REGEX REPLACE "(\r?\n)+$" "" PYTHON_USER_SITE_PACKAGE "${PYTHON_USER_SITE_PACKAGE}")
	set(PYMODULE_ROOT "${PYTHON_USER_SITE_PACKAGE}" CACHE PATH "PYMODULE_ROOT")
endif()

set(PYBIND11_CMAKE_DIR "/usr/local/share/cmake/pybind11" CACHE PATH "pybind11 CMake directory")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PYBIND11_CMAKE_DIR})
include(pybind11Tools)

add_subdirectory(${CMAKE_SOURCE_DIR}/libs)
add_subdirectory(${CMAKE_SOURCE_DIR}/projects)
add_subdirectory(${CMAKE_SOURCE_DIR}/test)
