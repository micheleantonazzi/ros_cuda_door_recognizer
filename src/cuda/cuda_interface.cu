#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_interface.h"
#include "utilities/gpu_utilities.h"
#include "../utilities/utilities.h"

__global__ void test_kernel(int num){
    for (int i = 0; i < 100; ++i) {
        printf("Hello fromm %i\n", num);
    }

}

void CudaInterface::test_cuda(){

    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
    for (int i = 0; i < num_streams; i++) {


        // launch one worker kernel per stream
        test_kernel<<<1, 64, 0, streams[0]>>>(i);

        // launch a dummy kernel on the default stream
        //test_kernel<<<1, 1, 0, 0>>>();
    }
    cudaDeviceReset();
}

Pixel* CudaInterface::getPixelArray(const unsigned char *imageData, int width, int height) {
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

    double time = Utilities::seconds();

    to_gray_scale<<<numBlocks, numThread>>>(destination, source, width, height);

    cudaDeviceSynchronize();

    time = Utilities::seconds() - time;

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

void CudaInterface::toGrayScale(Pixel *destination, Pixel *source, int width, int height, int numBlocks, int numThread, const cudaStream_t &stream) {

    to_gray_scale<<<numBlocks, numThread, 0, stream>>>(destination, source, width, height);
}

double CudaInterface::toGrayScale(Pixel *destination, Pixel *source, int width, int height, int numBlocks, int numThread) {
    double time = Utilities::seconds();

    to_gray_scale<<<numBlocks, numThread>>>(destination, source, width, height);

    cudaDeviceSynchronize();

    time = Utilities::seconds() - time;

    return time;
}


__constant__ float maskConstant[10];

__global__ void gaussian_filter_horizontal(Pixel *destination, Pixel *source, int width, int height,
                                int maskDim){
    extern __shared__ Pixel smem[];

    int pixelPerThread = (width * height) / (gridDim.x * blockDim.x) + 1;

    // First pixel of a block
    int blockStart = blockDim.x * pixelPerThread * blockIdx.x;

    // Load first values
    if (threadIdx.x < maskDim / 2 && blockStart + threadIdx.x < width * height){
        smem[threadIdx.x] = 0;
        int start = (blockStart + threadIdx.x) % width;
        if(start - maskDim / 2 >= 0)
            smem[threadIdx.x] = *(source + blockStart + threadIdx.x - maskDim / 2);
    }

    if(blockStart + threadIdx.x < width * height){
        for (int i = 0; i < pixelPerThread; ++i) {
            if(blockStart + (blockDim.x * i) + threadIdx.x < width * height){
                smem[maskDim / 2 + (blockDim.x * i) + threadIdx.x] = *(source + blockStart + (blockDim.x * i) + threadIdx.x);
            }
        }
    }

    // Load final part
    if(threadIdx.x >= blockDim.x - maskDim / 2 && blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x < width * height){
        smem[maskDim - 1 + blockDim.x * (pixelPerThread - 1) + threadIdx.x] = 0;
        if(blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x < width * height - maskDim / 2){
            smem[maskDim - 1 + blockDim.x * (pixelPerThread - 1) + threadIdx.x] = *(source + blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x + maskDim / 2);
        }
    }

    __syncthreads();

    for (int i = 0; i < pixelPerThread; ++i) {

        if(blockStart + (blockDim.x * i) + threadIdx.x < width * height){
            float value = 0;
            for (int j = 0; j < maskDim; ++j) {
                int column = (blockStart + (blockDim.x * i) + threadIdx.x) % width;
                if(column + j - maskDim / 2 >= 0 && column + j - maskDim / 2 < width){
                    unsigned char x = smem[j + (blockDim.x * i) + threadIdx.x];
                    value += x * maskConstant[j];
                }
            }
            int row = (blockStart + (blockDim.x * i) + threadIdx.x) / width;
            int column = (blockStart + (blockDim.x * i) + threadIdx.x) % width;
            unsigned char finalChar = value;
            float final = (finalChar << 16) + (finalChar << 8) + finalChar;
            *(destination + column * height + row) = final;
        }
    }
}


double CudaInterface::gaussianFilter(Pixel *destination, Pixel *source, int width, int height, float *gaussianMask,
                                     int maskDim, int numBlocks, int numThread) {
    // Alloc the constant memory
    cudaMemcpyToSymbol(maskConstant, gaussianMask, maskDim * sizeof(float));

    // Alloc device memory to put the transpose image
    Pixel *transposeImage;
    cudaMalloc(&transposeImage, width * height * sizeof(Pixel));

    int sharedMemory = ((width * height) / (numBlocks * numThread) + 1) * numThread + maskDim - 1;
    double time = Utilities::seconds();

    // Applying the first horizontal gaussian filter
    gaussian_filter_horizontal<<<numBlocks, numThread, sharedMemory * sizeof(Pixel)>>>(transposeImage, source, width, height, maskDim);
    // Applying the second horizontal gaussian filter
    gaussian_filter_horizontal<<<numBlocks, numThread, sharedMemory * sizeof(Pixel)>>>(destination, transposeImage, height, width, maskDim);
    cudaDeviceSynchronize();
    time = Utilities::seconds() - time;

    cudaFree(transposeImage);
    return time;
}

void CudaInterface::gaussianFilter(Pixel *destination, Pixel *source, int width, int height, float *gaussianMask,
                                     int maskDim, int numBlocks, int numThread, cudaStream_t &stream) {
    // Alloc the constant memory
    cudaMemcpyToSymbol(maskConstant, gaussianMask, maskDim * sizeof(float));

    // Alloc device memory to put the transpose image
    Pixel *transposeImage;
    cudaMalloc(&transposeImage, width * height * sizeof(Pixel));

    int sharedMemory = ((width * height) / (numBlocks * numThread) + 1) * numThread + maskDim - 1;

    // Applying the first horizontal gaussian filter
    gaussian_filter_horizontal<<<numBlocks, numThread, sharedMemory * sizeof(Pixel), stream>>>(transposeImage, source, width, height, maskDim);
    // Applying the second horizontal gaussian filter
    gaussian_filter_horizontal<<<numBlocks, numThread, sharedMemory * sizeof(Pixel), stream>>>(destination, transposeImage, height, width, maskDim);

    cudaStreamSynchronize(stream);
    cudaFree(transposeImage);
}