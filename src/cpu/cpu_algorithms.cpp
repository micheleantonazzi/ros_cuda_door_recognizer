//
// Created by michele on 07/07/19.
//

#include <stdint-gcc.h>
#include <sensor_msgs/Image.h>
#include "cpu_algorithms.h"
#include "../utilities/utilities.h"

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

void CpuAlgorithms::sobel(unsigned char *destination, unsigned char *source, int width, int height) {

    int maskDim = 3;
    int *sobelMaskHorizontal = Utilities::getSobelMaskHorizontal();

    float *sobelHorizontal = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = 0;
            for (int k = -maskDim / 2; k <= maskDim / 2; ++k) {
                for (int z = -maskDim / 2; z <= maskDim / 2; ++z) {
                    if(i + k >= 0 && i + k < height &&
                       j + z >= 0 && j + z < width) {
                        value += *(source + ((i + k) * width + j + z) * 3) * sobelMaskHorizontal[(k + maskDim / 2) * maskDim + (z + maskDim / 2)];
                    }
                }
            }

            *(sobelHorizontal + (i * width + j)) = value;
        }
    }

    int *sobelMaskVertical = Utilities::getSobelMaskVertical();
    float sobelVertical[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = 0;
            for (int k = -maskDim / 2; k <= maskDim / 2; ++k) {
                for (int z = -maskDim / 2; z <= maskDim / 2; ++z) {
                    if(i + k >= 0 && i + k < height &&
                       j + z >= 0 && j + z < width) {
                        value += *(source + ((i + k) * width + j + z) * 3) * sobelMaskVertical[(k + maskDim / 2) * maskDim + (z + maskDim / 2)];
                    }
                }
            }

            *(sobelVertical + (i * width + j)) = value;
        }
    }

    int direction[width * height];
    float gradient[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = sqrt(pow(sobelHorizontal[i * width + j], 2) + pow(sobelVertical[i * width + j], 2));
            *(gradient + i * width + j) = value;

            float dir = atan2(sobelVertical[i * width + j], sobelHorizontal[i * width + j]) * 180 / M_PI;
            if (dir < 0)
                dir += 180;
            if(dir > 22.5 && dir <= 67.5)
                dir = 45;
            else if(dir > 67.5 && dir <= 112.5)
                dir = 90;
            else if(dir > 112.5 && dir <= 157.5)
                dir = 135;
            else
                dir = 0;

            *(direction + i * width + j) = dir;
        }
    }

    float non_maximum_subpression[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int dir = *(direction + i * width + j);
            float first = 0;
            float second = 0;

            if(dir == 0){
                if(j - 1 >= 0)
                    first = *(gradient + i * width + j - 1);
                if(j + 1 < width)
                    second = *(gradient + i * width + j + 1);

            }
            else if(dir == 90){
                if(i - 1 >= 0)
                    first = *(gradient + (i - 1) * width + j);
                if(i + 1 < height)
                    second = *(gradient + (i + 1) * width + j);
            }
            else if(dir == 45){
                if(i - 1 >= 0 && j + 1 < width)
                    first = *(gradient + (i - 1) * width + j + 1);
                if(i + 1 < height && j - 1 >= 0)
                    second = *(gradient + (i + 1) * width + j - 1);
            }
            else if(dir == 135){
                if(i + 1 < height && j + 1 < width)
                    first = *(gradient + (i + 1) * width + j + 1);
                if(i - 1 >= 0 && j - 1 >= 0)
                    second = *(gradient + (i - 1) * width + j - 1);
            }

            float currentValue = *(gradient + i * width + j);

            if(!(currentValue >= first && currentValue >= second))
                *(non_maximum_subpression + i * width + j) = 0;
            else
                *(non_maximum_subpression + i * width + j) = currentValue;
        }
    }

    double average = 0;
    int div = 0;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = *(non_maximum_subpression + i * width + j);
            average += value;
            if(value > 0)
                div++;
            value = value > 50 ? 255 : 0;
            *(destination + (i * width + j) * 3) = value;
            *(destination + (i * width + j) * 3 + 1) = value;
            *(destination + (i * width + j) * 3 + 2)  = value;
        }
    }
    average /= div;
    printf("media %f\n", average);
}