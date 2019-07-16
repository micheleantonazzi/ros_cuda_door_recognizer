#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_interface.h"
#include "utilities/gpu_utilities.h"
#include "../utilities/time_utilities.h"

__global__ void test_kernel(){
    printf("Hello from\n");
}

void CudaInterface::test_cuda(){
    test_kernel<<<1, 10>>>();
    CHECK(cudaDeviceSynchronize());
}

Pixel24* CudaInterface::getPixelArray(unsigned char *imageData, int width, int height) {
    int imageSize = width * height;

    Pixel24 *pixelArray;

    cudaMallocHost(&pixelArray, imageSize * sizeof(Pixel24));

    for(int i = 0; i < imageSize; ++i){
        pixelArray[i].R = *(imageData++);
        pixelArray[i].G = *(imageData++);
        pixelArray[i].B = *(imageData++);
    }

    return pixelArray;

}

void CudaInterface::pixelArrayToCharArray(unsigned char *imageData, Pixel24 *source, int width, int height) {
    int imageSize = width * height;

    for (int i = 0; i < imageSize; ++i) {
        Pixel24 pixel = *(source++);
        *(imageData++) = pixel.R;
        *(imageData++) = pixel.G;
        *(imageData++) = pixel.B;
    }
}

__global__ void to_gray_scale(unsigned char *destination, unsigned char *source, int width, int height){
    int threadTot = gridDim.x * blockDim.x;

    int imageSize = width * height * 3;

    int valuesPerThread = (imageSize / threadTot) + 3;

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId * valuesPerThread < imageSize){

        // Move the pointer to the correct position
        source += threadId * valuesPerThread;
        destination += threadId * valuesPerThread;

        int start = threadId * valuesPerThread;

        for(int i = 0; i < valuesPerThread && start + i < imageSize; i += 3){
            unsigned char average = (*(source++) + *(source++) + *(source++)) / 3;
            *(destination++) = average;
            *(destination++) = average;
            *(destination++) = average;

        }
    }
}

double CudaInterface::toGrayScale(unsigned char *destination, unsigned char *source, int width, int height, int numBlocks, int numThread) {

    double time = seconds();

    to_gray_scale<<<numBlocks, numThread>>>(destination, source, width, height);

    cudaDeviceSynchronize();

    time = seconds() - time;

    return time;
}

__global__ void to_gray_scale(Pixel24 *destination, Pixel24 *source, int width, int height){

    int totThread = gridDim.x * blockDim.x;

    // Thread group is 32 (the warp dimension) if the total number of thread is equal or higher than warp dimension (32)
    int threadGroupDim = totThread >= 32 ? 32 : totThread;

    int imageSize = width * height;

    int jumpPerThreadGroup = (imageSize / totThread) + 1;

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    // The group of a thread
    int threadGroup = threadId / threadGroupDim;

    // The number if thread inside his group
    int threadIdInGroup = (blockDim.x * blockIdx.x + threadIdx.x) % threadGroupDim;

    if (threadGroupDim * threadGroup * jumpPerThreadGroup + threadIdInGroup < imageSize){

        // Move the pointer to the correct position
        // In this way the accesses to global memory are aligned and coalescent
        source += threadGroup * threadGroupDim * jumpPerThreadGroup + threadIdInGroup;
        destination += threadGroup * threadGroupDim * jumpPerThreadGroup + threadIdInGroup;

        int start = threadGroup * jumpPerThreadGroup * threadGroupDim + threadIdInGroup;

        for(int i = 0; i < jumpPerThreadGroup && start + i * threadGroupDim < imageSize; i++){

            source += threadGroupDim;
            destination += threadGroupDim;

            Pixel24 pixel24 = *source;
            unsigned char average = (pixel24.R + pixel24.G + pixel24.B) / 3;
            pixel24.R = average;
            pixel24.G = average;
            pixel24.B = average;
            *destination = pixel24;
        }
    }
}

double CudaInterface::toGrayScale(Pixel24 *destination, Pixel24 *source, int width, int height, int numBlocks, int numThread) {
    double time = seconds();

    to_gray_scale<<<numBlocks, numThread>>>(destination, source, width, height);

    cudaDeviceSynchronize();

    time = seconds() - time;

    return time;
}