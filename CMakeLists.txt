cmake_minimum_required(VERSION 3.15...3.31)
project(sxe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

pybind11_add_module(_core src/sxe/_core.cpp)
target_link_libraries(_core PRIVATE Threads::Threads)

# Portable optimisation only. -march=native is deliberately NOT used: a wheel
# built on one machine must run on every machine that installs it. Builders
# targeting a known host can opt in with -DSXE_NATIVE=ON.
if(MSVC)
  target_compile_options(_core PRIVATE /O2)
else()
  target_compile_options(_core PRIVATE -O3 -fno-math-errno)
endif()

option(SXE_NATIVE "Tune for the building machine's CPU (not redistributable)" OFF)
if(SXE_NATIVE AND NOT MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
  if(HAS_MARCH_NATIVE)
    target_compile_options(_core PRIVATE -march=native)
  endif()
endif()

install(TARGETS _core DESTINATION sxe)
