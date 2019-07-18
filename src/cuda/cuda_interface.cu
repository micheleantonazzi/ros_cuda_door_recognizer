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

    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

        // launch one worker kernel per stream
        test_kernel<<<1, 64, 0, streams[i]>>>();

        // launch a dummy kernel on the default stream
        test_kernel<<<1, 1, 0, 0>>>();
    }
    cudaDeviceReset();
}

Pixel* CudaInterface::getPixelArray(unsigned char *imageData, int width, int height) {
    int imageSize = width * height;

    Pixel *pixelArray;

    cudaMallocHost(&pixelArray, imageSize * sizeof(Pixel));

    for(int i = 0; i < imageSize; ++i)
        pixelArray[i] = (*(imageData++) << 16) + (*(imageData++) << 8) + *(imageData++);

    return pixelArray;

}

void CudaInterface::pixelArrayToCharArray(unsigned char *imageData, Pixel *source, int width, int height) {
    int imageSize = width * height;

    for (int i = 0; i < imageSize; ++i) {
        Pixel pixel = *(source++);
        *(imageData++) = pixel >> 16;
        *(imageData++) = pixel >> 8;
        *(imageData++) = pixel;
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

__global__ void to_gray_scale(Pixel *destination, Pixel *source, int width, int height){

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

            Pixel pixel = *source;

            unsigned char R = pixel >> 16;
            unsigned char G = pixel >> 8;
            unsigned char B = pixel;

            unsigned char average = (R + G + B) / 3;

            *destination = Pixel((average << 16) + (average << 8) + average);

            source += threadGroupDim;
            destination += threadGroupDim;
        }
    }
}

double CudaInterface::toGrayScale(Pixel *destination, Pixel *source, int width, int height, int numBlocks, int numThread) {
    double time = seconds();

    to_gray_scale<<<numBlocks, numThread>>>(destination, source, width, height);

    cudaDeviceSynchronize();

    time = seconds() - time;

    return time;
}