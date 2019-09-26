cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 14)
project(vmcore)

option(VMCORE_SHARED_LIBRARY "Set to ON to build using dynamic linking" ON)
if(VMCORE_SHARED_LIBRARY)
add_definitions(-DVMCORE_SHARED_LIBRARY)
set(VMCORE_SHARED "SHARED")
else()
set(VMCORE_SHARED "STATIC")
endif()


aux_source_directory(src/VMFoundation FOUNDATION_SRC)
aux_source_directory(src/VMGraphics GRAPHICS_SRC)

add_subdirectory(external/VMUtils)


add_library(vmfoundation ${VMCORE_SHARED} ${FOUNDATION_SRC})
add_dependencies(vmfoundation vmutils)
target_link_libraries(vmfoundation vmutils)
target_include_directories(vmfoundation PUBLIC "external/VMat/include" "external/VMUtils/include" "include" "external")


install(TARGETS vmfoundation LIBRARY DESTINATION "lib" RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
install(DIRECTORY "include" DESTINATION "include/VMutils")


add_library(vmgraphics ${VMCORE_SHARED} ${GRAPHICS_SRC})
add_dependencies(vmgraphics vmutils)
target_link_libraries(vmgraphics vmutils)
target_include_directories(vmgraphics PUBLIC "external/VMat/include" "external/VMUtils/include" "include" "external")

# submodule

install(TARGETS vmgraphics LIBRARY DESTINATION "lib" RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
install(DIRECTORY "include" DESTINATION "include/VMutils")
