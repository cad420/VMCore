#include <iostream>
#include <gtest/gtest.h>
#include <fstream>
#include <random>
#include <VMGraphics/interpulator.h>

TEST(test_color_interpulator, basic){

    vm::ColorInterpulator intp;

    std::string text = " \
        5 0.0 1.0\n \
        0.1 50 50 50 50 50 50 50 50\n \
        0.2 100 100 100 100 100 100 100 100\n\
        0.3 150 150 150 150 150 150 150 150\n\
        0.4 200 200 200 200 200 200 200 200\n\
        0.5 255 255 255 255 255 255 255 255\n\
    ";

    intp.ReadFromText(text);
    float color[256*4];
    intp.FetchData(color,256);
}