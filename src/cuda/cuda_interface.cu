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

__global__ void to_gray_scale(unsigned char *destination, unsigned char *source, int width, int height){
    int threadTot = gridDim.x * blockDim.x;

    int imageSize = width * height * 3;

    int pixelPerThread = (imageSize / threadTot) + 1;

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId * pixelPerThread < width * height * 3){

        // Move the pointer to the correct position
        source += threadId * pixelPerThread;
        destination += threadId * pixelPerThread;

        int limit = threadId * pixelPerThread;

        for(int i = 0; i < pixelPerThread && limit + i < imageSize; i += 3){
            unsigned char average = (*(source++) + *(source++) + *(source++)) / 3;
            *(destination++) = average;
            *(destination++) = average;
            *(destination++) = average;

        }
    }
}

double CudaInterface::toGrayScale(unsigned char *destination, unsigned char *source, int width, int height, int numBlocks, int numThread) {

    int sizeImage = width * height * 3;

    unsigned char *sourceGpu;
    unsigned char *destinationGpu;

    cudaMalloc(&sourceGpu, sizeof(unsigned char) * sizeImage);
    cudaMalloc(&destinationGpu, sizeof(unsigned char) * sizeImage);

    cudaMemcpy(sourceGpu, source, sizeImage * sizeof(unsigned char), cudaMemcpyHostToDevice);

    double time = seconds();

    to_gray_scale<<<numBlocks, numThread>>>(destinationGpu, sourceGpu, width, height);

    cudaDeviceSynchronize();

    time = seconds() - time;

    cudaMemcpy(destination, destinationGpu, sizeImage * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(sourceGpu);
    cudaFree(destinationGpu);

    return time;
}