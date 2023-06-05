cmake_minimum_required(VERSION 3.9)
project(vmcore)
set(CMAKE_CXX_STANDARD 17)

aux_source_directory(src/VMFoundation FOUNDATION_SRC)
aux_source_directory(src/VMGraphics GRAPHICS_SRC)

option(VMCORE_SHARED_LIBRARY "Set to ON to build using dynamic linking" OFF)
if (VMCORE_SHARED_LIBRARY)
  add_library(vmcore SHARED)
  target_compile_definitions(vmcore PUBLIC -DVMCORE_SHARED_LIBRARY)
else ()
  add_library(vmcore STATIC)
endif ()


include(FetchContent)

FetchContent_Declare(vmutil GIT_REPOSITORY https://github.com/cad420/VMUtils.git)
FetchContent_MakeAvailable(vmutil)

FetchContent_Declare(vmat GIT_REPOSITORY https://github.com/cad420/VMat.git)
FetchContent_MakeAvailable(vmat)

target_sources(vmcore PRIVATE ${GRAPHICS_SRC} ${FOUNDATION_SRC})
if(UNIX)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(vmcore PRIVATE VMat VMUtils Threads::Threads dl)
else()
target_link_libraries(vmcore PRIVATE VMat VMUtils)
endif()

target_include_directories(vmcore
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extension>
    $<INSTALL_INTERFACE:include>
)

option(VMCORE_BUILD_TEST "Set on to build test case" OFF)
if (VMCORE_BUILD_TEST)
add_subdirectory(test)
endif ()

option(VMCORE_INSTALL "install VMcore library & headers" OFF)
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



