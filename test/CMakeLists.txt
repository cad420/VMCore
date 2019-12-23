
cmake_minimum_required(VERSION 3.12)

project(test)


add_executable(test_rawreader)
target_sources(test_rawreader PRIVATE "test_rawreader.cpp")
target_link_libraries(test_rawreader vmcore)
target_include_directories(test_rawreader PRIVATE "../include" "../extension")

install(TARGETS test_rawreader LIBRARY DESTINATION "lib" RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")

add_executable(test_blockedgridvolumefile)
target_sources(test_blockedgridvolumefile PRIVATE "test_blockedgridvolumefile.cpp")
target_link_libraries(test_blockedgridvolumefile vmcore)
target_include_directories(test_blockedgridvolumefile PRIVATE "../include" "../extension")

install(TARGETS test_blockedgridvolumefile LIBRARY DESTINATION "lib" RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
