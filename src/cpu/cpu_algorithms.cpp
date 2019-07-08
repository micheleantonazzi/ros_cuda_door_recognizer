//
// Created by michele on 07/07/19.
//

#include <cstdio>
#include "cpu_algorithms.h"

CpuAlgorithms::CpuAlgorithms() {}

CpuAlgorithms& CpuAlgorithms::getInstance() {
    static CpuAlgorithms cpuAlgorithms;
    return cpuAlgorithms;
}

void CpuAlgorithms::toGrayScale(unsigned char *pixels, int width, int height) {
    int len = width * height * 3;
    for (int i = 0; i < len; i += 3) {
        unsigned char average = ( *(pixels) + *(pixels + 1) + *(pixels + 2) ) / 3;
        *(pixels++) = average;
        *(pixels++) = average;
        *(pixels++) = average;
    }
}