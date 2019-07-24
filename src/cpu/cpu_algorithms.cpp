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

void CpuAlgorithms::toGrayScale(uint8_t *destination, const uint8_t *source, int width, int height){
    int len = width * height * 3;

    for (int i = 0; i < len; i += 3) {
        unsigned char average = ( *(source++) + *(source++) + *(source++) ) / 3;
        *(destination++) = average;
        *(destination++) = average;
        *(destination++) = average;
    }
}

void CpuAlgorithms::copyArrayToImage(sensor_msgs::Image& destination, uint8_t *source){
    int len = destination.width * destination.height * 3;

    for (int i = 0; i < len; i += 3) {
        destination.data.push_back(*(source++));
        destination.data.push_back(*(source++));
        destination.data.push_back(*(source++));
    }
}

void CpuAlgorithms::gaussianFilter(unsigned char *destination, unsigned char *source, float *matrix, int width,
                                   int height, int matrixDim) {

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = 0;
            for (int k = -matrixDim / 2; k <= matrixDim / 2; ++k) {
                for (int z = -matrixDim / 2; z <= matrixDim / 2; ++z) {
                    if(i + k >= 0 && i + k < height &&
                            j + z >= 0 && j + z < width) {
                        value += *(source + ((i + k) * width + j + z) * 3) * matrix[(k + matrixDim / 2) * matrixDim + (z + matrixDim / 2)];
                    }
                }
            }
            *(destination + (i * width + j) * 3) = value;
            *(destination + (i * width + j) * 3 + 1) = value;
            *(destination + (i * width + j) * 3 + 2)  = value;
        }
    }
}