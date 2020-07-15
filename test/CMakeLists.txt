cmake_minimum_required(VERSION 3.12)

file(GLOB_RECURSE VMCORE_TEST_SRC *.cpp)
add_executable(vmcore_test_all)
target_sources(vmcore_test_all PRIVATE ${VMCORE_TEST_SRC})
target_link_libraries(vmcore_test_all vmcore)
target_include_directories(vmcore_test_all PRIVATE "../include" "../extension")

enable_testing()
find_package(GTest CONFIG REQUIRED)
target_link_libraries(vmcore_test_all GTest::gtest_main GTest::gtest GTest::gmock GTest::gmock_main)

include(GoogleTest)
gtest_add_tests(vmcore_test_all "" AUTO)
install(TARGETS vmcore_test_all LIBRARY DESTINATION "lib" RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
