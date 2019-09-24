cmake_minimum_required(VERSION 2.7)
project(vmutils)

file(GLOB_RECURSE SOURCES
  "${CMAKE_SOURCE_DIR}/src/*.cc"
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_library(vmutils ${SOURCES})
target_include_directories(vmutils PUBLIC "${CMAKE_SOURCE_DIR}/vmutils")

