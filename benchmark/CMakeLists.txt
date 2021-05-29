cmake_minimum_required(VERSION 3.12)

# becnmark_block3dcache.cpp
add_executable(vmcore_benchmark_block3dcache)
target_sources(vmcore_benchmark_block3dcache PRIVATE "benchmark_block3dcache.cpp")
target_link_libraries(vmcore_benchmark_block3dcache vmcore)
target_include_directories(vmcore_benchmark_block3dcache PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/extension")
install(TARGETS vmcore_benchmark_block3dcache LIBRARY DESTINATION "lib" RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")



install(TARGETS vmcore LIBRARY DESTINATION "lib" RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")