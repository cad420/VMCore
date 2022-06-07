cmake_minimum_required(VERSION 3.9)
project(vmcore)

aux_source_directory(src/VMFoundation FOUNDATION_SRC)
aux_source_directory(src/VMGraphics GRAPHICS_SRC)

option(VMCORE_SHARED_LIBRARY "Set to ON to build using dynamic linking" ON)
if (VMCORE_SHARED_LIBRARY)
  #add_definitions(-DVMCORE_SHARED_LIBRARY)
  add_library(vmcore SHARED)
  target_compile_definitions(vmcore PUBLIC -DVMCORE_SHARED_LIBRARY)
else ()
  add_library(vmcore STATIC)
endif ()
if (NOT MSVC)
  if (NOT APPLE)
    target_link_libraries(vmcore stdc++fs)
  endif()
endif ()

include(scripts/external.cmake)

vm_external_module(
    GIT_REPOSITORY https://github.com/cad420/VMat.git
    GIT_TAG master
)
vm_external_module(
    GIT_REPOSITORY https://github.com/cad420/VMUtils.git
    GIT_TAG master
)

vm_target_dependency(vmcore VMat PUBLIC)
vm_target_dependency(vmcore VMUtils PUBLIC)

target_sources(vmcore PRIVATE ${GRAPHICS_SRC} ${FOUNDATION_SRC} "test/test_memory.cpp")
if(UNIX)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(vmcore dl Threads::Threads) # for dlopen etc.
else()
target_link_libraries(vmcore)
endif()
target_include_directories(vmcore
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extension>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(vmcore PUBLIC cxx_std_17)

option(VMCORE_BUILD_TEST "Set on to build test case" ON)
if (VMCORE_BUILD_TEST)
add_subdirectory(test)
endif ()

option(VMCORE_BUILD_BENCHMARK "Set ON to build benchmark" ON)
if(VMCORE_BUILD_BENCHMARK)
add_subdirectory(benchmark)
endif ()

option(VMCORE_INSTALL "install VMcore library & headers" ON)
if (VMCORE_INSTALL)
  install(
      TARGETS vmcore
      DESTINATION lib
  )
  install(
      DIRECTORY
        include/VMFoundation
        include/VMGraphics
        extension/VMCoreExtension
      DESTINATION include
  )
endif()



