//
// Created by michele on 18/07/19.
//

#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "utilities.h"

double Utilities::seconds(){
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

float* Utilities::getGaussianMatrix(int size, float alpha) {
    float* matrix = new float[size * size];
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = (1 / (2 * M_PI * pow(alpha, 2)) *
                    pow(M_E, -((pow(i - size / 2, 2) + pow(j - size / 2, 2)) / (2 * pow(alpha, 2)))));
        }
    }

    return matrix;
}

float* Utilities::getGaussianMatrixPinned(int size, float alpha) {
    float* matrix;

    cudaMallocHost(&matrix, size * size * sizeof(float));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = (1 / (2 * M_PI * pow(alpha, 2)) *
                                    pow(M_E, -((pow(i - size / 2, 2) + pow(j - size / 2, 2)) / (2 * pow(alpha, 2)))));
        }
    }

    return matrix;
}