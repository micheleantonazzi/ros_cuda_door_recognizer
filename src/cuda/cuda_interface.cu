#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
    printf("Hello from \n");
}

double CudaInterface::toGrayScale(unsigned char *destination, unsigned char *source, int width, int height, int numBlocks, int numThread) {

    int sizeImage = width * height * 3;

    unsigned char *sourceGpu;
    unsigned char *destinationGpu;

    cudaMalloc(&sourceGpu, sizeof(unsigned char) * sizeImage);
    cudaMalloc(&destinationGpu, sizeof(unsigned char) * sizeImage);

    cudaMemcpy(sourceGpu, source, sizeImage, cudaMemcpyHostToDevice);

    double time = seconds();

    to_gray_scale<<<numBlocks, numThread>>>(destinationGpu, sourceGpu, width, height);

    time = seconds() - time;

    cudaMemcpy(destination, destinationGpu, sizeImage, cudaMemcpyDeviceToHost);

    cudaFree(sourceGpu);
    cudaFree(destinationGpu);

    return time;
}