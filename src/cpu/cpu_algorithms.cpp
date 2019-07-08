//
// Created by michele on 07/07/19.
//

#include <stdint-gcc.h>
#include <sensor_msgs/Image.h>
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

void CpuAlgorithms::toGrayScale(sensor_msgs::Image& destination, const sensor_msgs::Image& source){
    int len = source.width * source.height * 3;
    const uint8_t * s = source.data.data();

    for (int i = 0; i < len; i += 3) {
        unsigned char average = ( *(s++) + *(s++) + *(s++) ) / 3;
        destination.data.push_back(average);
        destination.data.push_back(average);
        destination.data.push_back(average);
    }
}