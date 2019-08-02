//
// Created by michele on 07/07/19.
//

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

void CpuAlgorithms::sobel(float *edgeGradient, int *edgeDirection, unsigned char *source, int width, int height){

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

            *(sobelHorizontal + i * width + j) = value;
        }
    }

    int *sobelMaskVertical = Utilities::getSobelMaskVertical();
    float *sobelVertical = new float[width * height];

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

            *(sobelVertical + i * width + j) = value;
        }
    }

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = sqrt(pow(sobelHorizontal[i * width + j], 2) + pow(sobelVertical[i * width + j], 2));
            *(edgeGradient + i * width + j) = value;
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

            *(edgeDirection + i * width + j) = dir;
        }
    }

    delete(sobelMaskHorizontal);
    delete(sobelMaskVertical);
    delete(sobelHorizontal);
    delete(sobelVertical);
}

void CpuAlgorithms::harris(unsigned char *destination, unsigned char *source, unsigned char *imageSobel, int width, int height) {
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

            *(sobelHorizontal + i * width + j) = value;
        }
    }

    int *sobelMaskVertical = Utilities::getSobelMaskVertical();
    float *sobelVertical = new float[width * height];

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

            *(sobelVertical + i * width + j) = value;
        }
    }

    double time = Utilities::seconds();

    float *sobelHorizontal2 = new float[width * height];
    float *sobelVertical2 = new float[width * height];
    float *sobelHorizontalVertical = new float[width * height];

    // corner detection
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float x = *(sobelHorizontal + i * width + j);
            float y = *(sobelVertical + i * width + j);

            *(sobelHorizontal2 + i * width + j) = x * x;
            *(sobelVertical2 + i * width + j) = y * y;
            *(sobelHorizontalVertical + i * width + j) = x * y;
        }
    }

    float *sobelHorizontal2Sum = new float[width * height];
    float *sobelVertical2Sum = new float[width * height];
    float *sobelHorizontalVerticalSum = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float valueX = 0;
            float valueY = 0;
            float valueXY = 0;
            for (int k = -1; k <= 1; ++k) {
                for (int z = -1; z <= 1; ++z) {
                    if(i + k >= 0 && i + k < height &&
                       j + z >= 0 && j + z < width) {
                        valueX += *(sobelHorizontal2 + ((i + k) * width + j + z));
                        valueY += *(sobelVertical2 + ((i + k) * width + j + z));
                        valueXY += *(sobelHorizontalVertical + ((i + k) * width + j + z));
                    }
                }
            }

            *(sobelHorizontal2Sum + i * width + j) = valueX;
            *(sobelVertical2Sum + i * width + j) = valueY;
            *(sobelHorizontalVerticalSum + i * width + j) = valueXY;
        }
    }

    float *corners = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float x = *(sobelHorizontal2Sum + i * width + j);
            float y = *(sobelVertical2Sum + i * width + j);
            float xy = *(sobelHorizontalVerticalSum + i * width + j);

            *(corners + i * width + j) = (x * y - xy * xy) - 0.06 * ((x + y) * (x + y));
        }
    }

    float *cornerSuppressed = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float first = 0;
            float second = 0;
            float max = true;
            float currentValue = *(corners + i * width + j);

            /*for (int k = -1; k <= 1 && max; ++k) {
                for (int z = -1; z <= 1 && max; ++z) {
                    if(i + k >= 0 && i + k < height &&
                       j + z >= 0 && j + z < width) {
                        if(currentValue < *(corners + ((i + k) * width + j + z)))
                            max = false;

                    }
                }
            }
            */

            if(currentValue > 10000){
                if(!max)
                    currentValue = 0;
                else
                    currentValue = 255;
            } else
                currentValue = 0;


            *(cornerSuppressed + (i * width + j)) = currentValue;
        }
    }

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float r = *(cornerSuppressed + i * width + j);
            unsigned  char final;
            if(r == 0){
                final = *(imageSobel + (i * width + j)*3);
                *(destination + (i * width + j) * 3) = final;
                *(destination + (i * width + j) * 3 + 1) = final;
                *(destination + (i * width + j) * 3 + 2) = final;
            }

            else{
                *(destination + (i * width + j) * 3) = 0;
                *(destination + (i * width + j) * 3 + 1) = 255;
                *(destination + (i * width + j) * 3 + 2) = 0;
            }

        }
    }

    delete(sobelMaskHorizontal);
    delete(sobelMaskVertical);
    delete(sobelHorizontal);
    delete(sobelVertical);
    delete(cornerSuppressed);
    delete(corners);
    delete(sobelHorizontal2);
    delete(sobelHorizontal2Sum);
    delete(sobelVertical2);
    delete(sobelVertical2Sum);
    delete(sobelHorizontalVertical);
    delete(sobelHorizontalVerticalSum);
}

void CpuAlgorithms::nonMaximumSuppression(unsigned char *destination, float *edgeGradient, int *edgeDirection, int width, int height) {

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int dir = *(edgeDirection + i * width + j);
            float first = 0;
            float second = 0;

            if(dir == 0){
                if(j - 1 >= 0)
                    first = *(edgeGradient + i * width + j - 1);
                if(j + 1 < width)
                    second = *(edgeGradient + i * width + j + 1);
            }
            else if(dir == 90){
                if(i - 1 >= 0)
                    first = *(edgeGradient + (i - 1) * width + j);
                if(i + 1 < height)
                    second = *(edgeGradient + (i + 1) * width + j);
            }
            else if(dir == 45){
                if(i - 1 >= 0 && j + 1 < width)
                    first = *(edgeGradient + (i - 1) * width + j + 1);
                if(i + 1 < height && j - 1 >= 0)
                    second = *(edgeGradient + (i + 1) * width + j - 1);
            }
            else if(dir == 135){
                if(i + 1 < height && j + 1 < width)
                    first = *(edgeGradient + (i + 1) * width + j + 1);
                if(i - 1 >= 0 && j - 1 >= 0)
                    second = *(edgeGradient + (i - 1) * width + j - 1);
            }

            float currentValue = *(edgeGradient + i * width + j);

            if(!(currentValue >= first && currentValue >= second))
                currentValue = 0;
            else if(currentValue > 50)
                currentValue = 255;
            else
                currentValue = 0;

            *(destination + (i * width + j) * 3) = currentValue;
            *(destination + (i * width + j) * 3 + 1) = currentValue;
            *(destination + (i * width + j) * 3 + 2)  = currentValue;
        }
    }
}