cmake_minimum_required(VERSION 3.18)
project(rewardhackbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(_core cpp/src/bindings.cpp)
target_include_directories(_core PRIVATE cpp/include)
target_link_libraries(_core PRIVATE Threads::Threads)
target_compile_options(_core PRIVATE -Wall -Wextra -O3)

# Drop the built module next to the Python package so no install step is needed.
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/rewardhackbench)
