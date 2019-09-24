cmake_minimum_required(VERSION 2.7)
set(CMAKE_CXX_STANDARD 14)
project(vmutils)

file(GLOB_RECURSE SOURCES
  src/*.cc
  src/*.cpp
)

add_library(vmutils ${SOURCES})
target_include_directories(vmutils PUBLIC vmutils)

