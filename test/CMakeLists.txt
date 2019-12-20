
cmake_minimum_required(VERSION 3.12)

project(test)


add_executable(test)
target_sources(test PRIVATE "test.cpp")
target_link_libraries(test vmcore)
target_include_directories(test PRIVATE "../include" "../extension")

install(TARGETS test LIBRARY DESTINATION "lib" RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
