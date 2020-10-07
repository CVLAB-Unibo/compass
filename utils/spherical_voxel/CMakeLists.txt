cmake_minimum_required(VERSION 2.8.12)
project(spherical_voxel)
set (CMAKE_CXX_STANDARD 11)

find_package(pybind11 REQUIRED)

set(SOURCE_DIR "src")
set(srcs "${SOURCE_DIR}/spherical_voxel.cc")

pybind11_add_module(spherical_voxel ${srcs})
