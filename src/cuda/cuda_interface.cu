#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
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
        *(pixelArray + i) = (*(imageData++) << 16) + (*(imageData++) << 8) + *(imageData++);

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


__constant__ float maskConstant[5];

__global__ void gaussian_filter_horizontal(Pixel *destination, Pixel *source, int width, int height,
                                int maskDim){
    extern __shared__ float smem[];

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
        for (int i = 0; i < pixelPerThread && blockStart + (blockDim.x * i) + threadIdx.x < width * height; ++i) {
            smem[maskDim / 2 + (blockDim.x * i) + threadIdx.x] = *(source + blockStart + (blockDim.x * i) + threadIdx.x);
        }
    }

    // Load final part
    if(threadIdx.x >= blockDim.x - maskDim / 2 && blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x < width * height){
        smem[maskDim - 1 + blockDim.x * (pixelPerThread - 1) + threadIdx.x] = 0;
        if((blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x) % width + maskDim / 2 < width){
            smem[maskDim - 1 + blockDim.x * (pixelPerThread - 1) + threadIdx.x] = *(source + blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x + maskDim / 2);
        }
    }

    __syncthreads();

    for (int i = 0; i < pixelPerThread && blockStart + (blockDim.x * i) + threadIdx.x < width * height; ++i) {

        float value = 0;
        for (int j = 0; j < maskDim; ++j) {
            int column = (blockStart + (blockDim.x * i) + threadIdx.x) % width;
            if(column + j - maskDim / 2 >= 0 && column + j - maskDim / 2 < width){
                int pixel = smem[j + (blockDim.x * i) + threadIdx.x];
                value += ((unsigned char) pixel) * maskConstant[j];
            }
        }
        int row = (blockStart + (blockDim.x * i) + threadIdx.x) / width;
        int column = (blockStart + (blockDim.x * i) + threadIdx.x) % width;
        unsigned char finalChar = value;
        int final = (finalChar << 16) + (finalChar << 8) + finalChar;
        *(destination + column * height + row) = final;

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

void CudaInterface::gaussianFilter(Pixel *destination, Pixel *source, Pixel *transposeImage, int width, int height, float *gaussianMask,
                                     int maskDim, int numBlocks, int numThread, cudaStream_t &stream) {
    // Alloc the constant memory
    cudaMemcpyToSymbolAsync(maskConstant, gaussianMask, maskDim * sizeof(float), 0, cudaMemcpyHostToDevice, stream);

    int sharedMemory = ((width * height) / (numBlocks * numThread) + 1) * numThread + maskDim - 1;

    // Applying the first horizontal gaussian filter
    gaussian_filter_horizontal<<<numBlocks, numThread, sharedMemory * sizeof(Pixel), stream>>>(transposeImage, source, width, height, maskDim);
    // Applying the second horizontal gaussian filter
    gaussian_filter_horizontal<<<numBlocks, numThread, sharedMemory * sizeof(Pixel), stream>>>(destination, transposeImage, height, width, maskDim);
}

// The sobel filter requires two bi-dimensional convolution, one horizontal and one vertical
// My implementation applies four one-dimensional convolution, two horizontal and two vertical
// Each group of two convolution is applied by two different kernel.
// This kernel are very similar but the first accepts a Pixel image and convert it in a float image
// in order to represent the edge gradient

__constant__ int sobelKernel1[3] = {-1, 0, 1};
__constant__ int sobelKernel2[3] = {1, 2, 1};

__global__ void sobel_convolution_one(float *destination, Pixel *source, int constant, int width, int height){

    int *kernel = (int *) &sobelKernel1;
    if(constant == 2)
        kernel = (int *) &sobelKernel2;

    extern __shared__ float smem[];

    int pixelPerThread = (width * height) / (gridDim.x * blockDim.x) + 1;

    // First pixel of a block
    int blockStart = blockDim.x * pixelPerThread * blockIdx.x;

    // Load first values
    if (threadIdx.x < 1 && blockStart + threadIdx.x < width * height){
        smem[threadIdx.x] = 0;
        int start = (blockStart + threadIdx.x) % width;
        if(start - 1 >= 0)
            smem[threadIdx.x] = *(source + blockStart + threadIdx.x - 1);
    }

    if(blockStart + threadIdx.x < width * height){
        for (int i = 0; i < pixelPerThread; ++i) {
            if(blockStart + (blockDim.x * i) + threadIdx.x < width * height){
                smem[1 + (blockDim.x * i) + threadIdx.x] = *(source + blockStart + (blockDim.x * i) + threadIdx.x);
            }
        }
    }

    // Load final part
    if(threadIdx.x >= blockDim.x - 1 && blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x < width * height){
        smem[2 + blockDim.x * (pixelPerThread - 1) + threadIdx.x] = 0;
        if(blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x < width * height - 1){
            smem[2 + blockDim.x * (pixelPerThread - 1) + threadIdx.x] = *(source + blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x + 1);
        }
    }

    __syncthreads();

    for (int i = 0; i < pixelPerThread && blockStart + (blockDim.x * i) + threadIdx.x < width * height; ++i) {

        float value = 0;
        for (int j = 0; j < 3; ++j) {
            int column = (blockStart + (blockDim.x * i) + threadIdx.x) % width;
            if(column + j - 1 >= 0 && column + j - 1 < width){
                float pixel = smem[j + (blockDim.x * i) + threadIdx.x];
                value += (unsigned char) pixel * kernel[j];
            }
        }
        int row = (blockStart + (blockDim.x * i) + threadIdx.x) / width;
        int column = (blockStart + (blockDim.x * i) + threadIdx.x) % width;
        *(destination + column * height + row) = value;
    }
}

__global__ void sobel_convolution_two(float *destination, float *source, int constant, int width, int height){

    int *kernel = (int *) &sobelKernel1;
    if(constant == 2)
        kernel = (int *) &sobelKernel2;

    extern __shared__ float smem[];

    int pixelPerThread = (width * height) / (gridDim.x * blockDim.x) + 1;

    // First pixel of a block
    int blockStart = blockDim.x * pixelPerThread * blockIdx.x;

    // Load first values
    if (threadIdx.x < 1 && blockStart + threadIdx.x < width * height){
        smem[threadIdx.x] = 0;
        int start = (blockStart + threadIdx.x) % width;
        if(start - 1 >= 0)
            smem[threadIdx.x] = *(source + blockStart + threadIdx.x - 1);
    }

    if(blockStart + threadIdx.x < width * height){
        for (int i = 0; i < pixelPerThread; ++i) {
            if(blockStart + (blockDim.x * i) + threadIdx.x < width * height){
                smem[1 + (blockDim.x * i) + threadIdx.x] = *(source + blockStart + (blockDim.x * i) + threadIdx.x);
            }
        }
    }

    // Load final part
    if(threadIdx.x >= blockDim.x - 1 && blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x < width * height){
        smem[2 + blockDim.x * (pixelPerThread - 1) + threadIdx.x] = 0;
        if(blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x < width * height - 1){
            smem[2 + blockDim.x * (pixelPerThread - 1) + threadIdx.x] = *(source + blockStart + blockDim.x * (pixelPerThread - 1) + threadIdx.x + 1);
        }
    }

    __syncthreads();

    for (int i = 0; i < pixelPerThread && blockStart + (blockDim.x * i) + threadIdx.x < width * height; ++i) {

        float value = 0;
        for (int j = 0; j < 3; ++j) {
            int column = (blockStart + (blockDim.x * i) + threadIdx.x) % width;
            if(column + j - 1 >= 0 && column + j - 1 < width){
                float pixel = smem[j + (blockDim.x * i) + threadIdx.x];
                value += pixel * kernel[j];
            }
        }
        int row = (blockStart + (blockDim.x * i) + threadIdx.x) / width;
        int column = (blockStart + (blockDim.x * i) + threadIdx.x) % width;
        *(destination + column * height + row) = value;
    }
}

__global__ void edge_gradient_direction(float *edgeGradient, int *edgeDirection, float *sobelHorizontal, float *sobelVertical, int width, int height) {
    int imageSize = width * height;

    int threadTot = gridDim.x * blockDim.x;

    int valuesPerThread = (imageSize / threadTot) + 1;

    int start = blockDim.x * blockIdx.x * valuesPerThread;

    for (int i = 0; i < valuesPerThread && start + i * blockDim.x + threadIdx.x < imageSize; i++) {
        int pos = start + i * blockDim.x + threadIdx.x;
        float x = sobelHorizontal[pos];
        float y = sobelVertical[pos];

        // The function pow isn't used to improve the performance, in fact the float operation is faster than the double ones
        float gradient = sqrt(x * x + y * y);
        edgeGradient[pos] = gradient;

        // All values are cart to float in order to improve the performance, in fact the float operation is faster than the double ones
        float dir = (float) atan2(y, x) * 180 / (float) M_PI;

        if (dir < 0)
            dir += 180;
        if (dir > 22.5 && dir <= 67.5)
            dir = 45;
        else if (dir > 67.5 && dir <= 112.5)
            dir = 90;
        else if (dir > 112.5 && dir <= 157.5)
            dir = 135;
        else
            dir = 0;
        edgeDirection[pos] = dir;
    }
}

__constant__ int sobelKernelHorizontal[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int sobelKernelVertical[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

__global__ void sobel_convolution(float *destination, Pixel *source, int type, int width, int height){
    int *kernel = (int *) &sobelKernelHorizontal;
    if(type == 2)
        kernel = (int *) &sobelKernelVertical;

    extern __shared__ float smem[];

    int pixelPerThread = (width * height) / (gridDim.x * blockDim.x) + 1;

    // First pixel of a block
    int blockStart = blockDim.x * pixelPerThread * blockIdx.x;

    // Load first values
    if (threadIdx.x < 3 && blockStart + width * ((int) threadIdx.x - 1) < width * height && blockStart + width * ((int) threadIdx.x - 1) >= 0){

        smem[(pixelPerThread * blockDim.x + 2) * threadIdx.x] = 0;
        if(blockStart / width + ((int) threadIdx.x - 1) >= 0 && blockStart / width + ((int) threadIdx.x - 1) < height &&
           blockStart % width - 1 >= 0){
            float value = *(source + blockStart + width * ((int) threadIdx.x - 1) - 1);
            value = (unsigned char) value;
            smem[(pixelPerThread * blockDim.x + 2) * threadIdx.x] = value;
        }
    }

    if(blockStart + threadIdx.x < width * height){
        for (int i = 0; i < pixelPerThread && blockStart + (blockDim.x * i) + threadIdx.x < width * height; ++i) {
            for(int y = -1; y < 2; y++){
                if(blockStart + (blockDim.x * i) + threadIdx.x + width * y  < width * height &&
                   (int) blockStart + ((int) blockDim.x * i) + (int) threadIdx.x + width * y  >= 0){
                    float value = *(source + blockStart + (blockDim.x * i) + threadIdx.x + width * y);
                    value = (unsigned char) value;
                    smem[(1 + (blockDim.x * i) + threadIdx.x) + (blockDim.x * pixelPerThread + 2) * (y + 1)] = value;
                }
            }
        }
    }

    // Load final part
    if(threadIdx.x >= blockDim.x - 3){
        int threadId = ((int) threadIdx.x + 3) % (int) blockDim.x;
        int value = blockStart + blockDim.x * pixelPerThread - 1 + width * (threadId - 1);
        if(value < width * height && value >= 0){
            smem[2 + blockDim.x * pixelPerThread - 1 + (2 + blockDim.x * pixelPerThread) * threadId] = 0;
            int start = blockStart + blockDim.x * pixelPerThread - 1;
            if(start / width + (threadId - 1) >= 0 && start / width + (threadId - 1) < height && start % width + 1 < width) {
                float pixel = *(source + value);
                pixel = (unsigned char) pixel;
                smem[2 + blockDim.x * pixelPerThread - 1 + (2 + blockDim.x * pixelPerThread) * threadId] = pixel;
            }
        }
    }

    __syncthreads();

    for (int i = 0; i < pixelPerThread && blockStart + (blockDim.x * i) + threadIdx.x < width * height; ++i) {

        float value = 0;
        for (int j = -1; j < 2; ++j) {
            for(int z = -1; z < 2; z++){
                int row = (blockStart + (blockDim.x * i) + threadIdx.x) / width;
                int column = (blockStart + (blockDim.x * i) + threadIdx.x) % width;
                if(column + z >= 0 && column + z < width && row + j >= 0 && row + j < height){
                    float pixel = smem[(blockDim.x * i) + threadIdx.x + (j + 1) * (2 + blockDim.x * pixelPerThread) + (z + 1)];
                    value += pixel * kernel[(j + 1) * 3 + z + 1];
                }
            }
        }

        *(destination + blockStart + (blockDim.x * i) + threadIdx.x) = value;
    }
}



double CudaInterface::sobelFilter(float *edgeGradient, int *edgeDirection, Pixel *source, int width, int height,
                                  int numBlocksConvolutionTwoDim, int numThreadConvolutionTwoDim,
                                  int numBlockLinear, int numThreadLinear) {

    float *sobelHorizontal, *sobelVertical;

    cudaMalloc(&sobelHorizontal, width * height * sizeof(float));
    cudaMalloc(&sobelVertical, width * height * sizeof(float));

    int sharedMemory = ((width * height) / (numBlocksConvolutionTwoDim * numThreadConvolutionTwoDim) + 1) * numThreadConvolutionTwoDim * 3 + 6;

    double time = Utilities::seconds();
    // Horizontal convolution
    sobel_convolution<<<numBlocksConvolutionTwoDim, numThreadConvolutionTwoDim, sharedMemory * sizeof(Pixel)>>>(sobelHorizontal, source, 1, width, height);
    // Vertical convolution
    sobel_convolution<<<numBlocksConvolutionTwoDim, numThreadConvolutionTwoDim, sharedMemory * sizeof(Pixel)>>>(sobelVertical, source, 2, width, height);

    edge_gradient_direction<<<numBlockLinear, numThreadLinear>>>(edgeGradient, edgeDirection, sobelHorizontal, sobelVertical, width, height);
    cudaDeviceSynchronize();
    time = Utilities::seconds() - time;

    cudaFree(sobelHorizontal);
    cudaFree(sobelVertical);
    return time;
}

void CudaInterface::sobelFilter(float *edgeGradient, int *edgeDirection, Pixel *source, float *sobelHorizontal,
                                float *sobelVertical, int width, int height, int numBlocksConvolutionTwoDim, int numThreadConvolutionTwoDim,
                                int numBlockLinear, int numThreadLinear, cudaStream_t &stream) {

    int sharedMemory = ((width * height) / (numBlocksConvolutionTwoDim * numThreadConvolutionTwoDim) + 1) * numThreadConvolutionTwoDim * 3 + 6;

    // Horizontal convolution
    sobel_convolution<<<numBlocksConvolutionTwoDim, numThreadConvolutionTwoDim, sharedMemory * sizeof(Pixel), stream>>>(sobelHorizontal, source, 1, width, height);
    // Vertical convolution
    sobel_convolution<<<numBlocksConvolutionTwoDim, numThreadConvolutionTwoDim, sharedMemory * sizeof(Pixel), stream>>>(sobelVertical, source, 2, width, height);

    edge_gradient_direction<<<numBlockLinear, numThreadLinear, 0, stream>>>(edgeGradient, edgeDirection, sobelHorizontal, sobelVertical, width, height);
}


__global__ void non_maximum_suppression(Pixel *destination, float *edgeGradient, int *edgeDirection, int width, int height) {
    int imageSize = width * height;

    int threadTot = gridDim.x * blockDim.x;

    int valuesPerThread = (imageSize / threadTot) + 1;

    int start = blockDim.x * blockIdx.x * valuesPerThread;

    for (int i = 0; i < valuesPerThread && start + i * blockDim.x + threadIdx.x < imageSize; i++) {
        int pos = start + i * blockDim.x + threadIdx.x;
        int x = pos / width;
        int y = pos % width;

        int dir = edgeDirection[pos];
        float first = 0;
        float second = 0;

        if (dir == 0) {
            if (y - 1 >= 0)
                first = *(edgeGradient + x * width + y - 1);
            if (y + 1 < width)
                second = *(edgeGradient + x * width + y + 1);
        } else if (dir == 90) {
            if (x - 1 >= 0)
                first = *(edgeGradient + (x - 1) * width + y);
            if (x + 1 < height)
                second = *(edgeGradient + (x + 1) * width + y);
        } else if (dir == 45) {
            if (x - 1 >= 0 && y + 1 < width)
                first = *(edgeGradient + (x - 1) * width + y + 1);
            if (x + 1 < height && y - 1 >= 0)
                second = *(edgeGradient + (x + 1) * width + y - 1);
        } else if (dir == 135) {
            if (x + 1 < height && y + 1 < width)
                first = *(edgeGradient + (x + 1) * width + y + 1);
            if (x - 1 >= 0 && y - 1 >= 0)
                second = *(edgeGradient + (x - 1) * width + y - 1);
        }

        float currentValue = edgeGradient[pos];

        if (!(currentValue >= first && currentValue >= second))
            currentValue = 0;
        unsigned char finalChar = 0;

        if (currentValue > 50)
            finalChar = 255;

        float final = (finalChar << 16) + (finalChar << 8) + finalChar;
        destination[pos] = final;
    }
}

double CudaInterface::nonMaximumSuppression(Pixel *destination, float *edgeGradient, int *edgeDirection, int width,
                                            int height, int numBlocks, int numThread) {
    double time = Utilities::seconds();
    non_maximum_suppression<<<numBlocks, numThread>>>(destination, edgeGradient, edgeDirection, width, height);
    cudaDeviceSynchronize();
    return Utilities::seconds() - time;
}

void CudaInterface::nonMaximumSuppression(Pixel *destination, float *edgeGradient, int *edgeDirection, int width,
                                            int height, int numBlocks, int numThread, cudaStream_t &stream) {
    non_maximum_suppression<<<numBlocks, numThread, 0, stream>>>(destination, edgeGradient, edgeDirection, width, height);
}

__global__ void harris_matrix(float *sobelHorizontal, float *sobelVertical, float *sobelHorizontalVertical, int width, int height){
    int imageSize = width * height;

    int threadTot = gridDim.x * blockDim.x;

    int valuesPerThread = (imageSize / threadTot) + 1;

    int start = blockDim.x * blockIdx.x * valuesPerThread;

    for (int i = 0; i < valuesPerThread && start + i * blockDim.x + threadIdx.x < imageSize; i++) {
        int pos = start + i * blockDim.x + threadIdx.x;
        float x = *(sobelHorizontal + pos);
        float y = *(sobelVertical + pos);

        *(sobelHorizontalVertical + pos) = x * y;
        *(sobelHorizontal + pos) = x * x;
        *(sobelVertical + pos) = y * y;
    }
}

// This kernel sums values inside a 3x3 kernel: it's a 2D convolution
__global__ void harris_matrix_sum(float *matrixSum, float *matrix, int width, int height){

    extern __shared__ float smem[];

    int pixelPerThread = (width * height) / (gridDim.x * blockDim.x) + 1;

    // First pixel of a block
    int blockStart = blockDim.x * pixelPerThread * blockIdx.x;

    // Load first values
    if (threadIdx.x < 3 && blockStart + width * ((int) threadIdx.x - 1) < width * height && blockStart + width * ((int) threadIdx.x - 1) >= 0){

        smem[(pixelPerThread * blockDim.x + 2) * threadIdx.x] = 0;
        if(blockStart / width + ((int) threadIdx.x - 1) >= 0 && blockStart / width + ((int) threadIdx.x - 1) < height &&
            blockStart % width - 1 >= 0){
            smem[(pixelPerThread * blockDim.x + 2) * threadIdx.x] = *(matrix + blockStart + width * ((int) threadIdx.x - 1) - 1);
        }
    }

    if(blockStart + threadIdx.x < width * height){
        for (int i = 0; i < pixelPerThread && blockStart + (blockDim.x * i) + threadIdx.x < width * height; ++i) {
            for(int y = -1; y < 2; y++){
                if(blockStart + (blockDim.x * i) + threadIdx.x + width * y  < width * height &&
                        (int) blockStart + ((int) blockDim.x * i) + (int) threadIdx.x + width * y  >= 0){
                    smem[(1 + (blockDim.x * i) + threadIdx.x) + (blockDim.x * pixelPerThread + 2) * (y + 1)] =
                            *(matrix + blockStart + (blockDim.x * i) + threadIdx.x + width * y);
                }
            }
        }
    }

    // Load final part
    if(threadIdx.x >= blockDim.x - 3){
        int threadId = ((int) threadIdx.x + 3) % (int) blockDim.x;
        int value = blockStart + blockDim.x * pixelPerThread - 1 + width * (threadId - 1);
        if(value < width * height && value >= 0){
            smem[2 + blockDim.x * pixelPerThread - 1 + (2 + blockDim.x * pixelPerThread) * threadId] = 0;
            int start = blockStart + blockDim.x * pixelPerThread - 1;
            if(start / width + (threadId - 1) >= 0 && start / width + (threadId - 1) >= 0 && start % width + 1 < width) {
                smem[2 + blockDim.x * pixelPerThread - 1 + (2 + blockDim.x * pixelPerThread) * threadId] = *(matrix + value);
            }
        }
    }

    __syncthreads();

    for (int i = 0; i < pixelPerThread && blockStart + (blockDim.x * i) + threadIdx.x < width * height; ++i) {

        float value = 0;
        for (int j = -1; j < 2; ++j) {
            for(int z = -1; z < 2; z++){
                int row = (blockStart + (blockDim.x * i) + threadIdx.x) / width;
                int column = (blockStart + (blockDim.x * i) + threadIdx.x) % width;
                if(column + z >= 0 && column + z < width && row + j >= 0 && row + j < height){
                    float pixel = smem[(blockDim.x * i) + threadIdx.x + (j + 1) * (2 + blockDim.x * pixelPerThread) + (z + 1)];
                    value += pixel;
                }
            }
        }

        *(matrixSum + blockStart + (blockDim.x * i) + threadIdx.x) = value;
    }
}

__global__ void harris_final_combination(Pixel *destination, float *sobelHorizontalSum, float *sobelVerticalSum,
        float *sobelHorizontalVerticalSum, int width, int height){

    int imageSize = width * height;

    int threadTot = gridDim.x * blockDim.x;

    int valuesPerThread = (imageSize / threadTot) + 1;

    int start = blockDim.x * blockIdx.x * valuesPerThread;

    for (int i = 0; i < valuesPerThread && start + i * blockDim.x + threadIdx.x < imageSize; i++) {
        int pos = start + i * blockDim.x + threadIdx.x;
        float x = *(sobelHorizontalSum + pos);
        float y = *(sobelVerticalSum + pos);
        float xy = *(sobelHorizontalVerticalSum + pos);

        float value = (x * y - xy * xy) - 0.06f * ((x + y) * (x + y));

        if(value > 9000000)
            *(destination + pos) = 255;
    }
}

__global__ void print(float *matrix, int width, int height){
    for (int i = height - 3; i < height; ++i) {
        for (int j = width - 3; j < width; ++j) {
            printf("%f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

double CudaInterface::harris(Pixel *destination, Pixel *source, int width, int height,
        int numBlocksTwoDimConvolution, int numThreadTwoDimConvolution, int numBlockLinear, int numThreadLinear) {

    float *sobelHorizontal, *sobelVertical, *sobelHorizontalVertical,
            *sobelHorizontalSum, *sobelVerticalSum, *sobelHorizontalVerticalSum;

    cudaMalloc(&sobelHorizontal, width * height * sizeof(float));
    cudaMalloc(&sobelVertical, width * height * sizeof(float));
    cudaMalloc(&sobelHorizontalVertical, width * height * sizeof(float));
    cudaMalloc(&sobelHorizontalSum, width * height * sizeof(float));
    cudaMalloc(&sobelVerticalSum, width * height * sizeof(float));
    cudaMalloc(&sobelHorizontalVerticalSum, width * height * sizeof(float));

    int sharedMemory = ((width * height) / (numBlocksTwoDimConvolution * numThreadTwoDimConvolution) + 1) * numThreadTwoDimConvolution * 3 + 6;

    double time = Utilities::seconds();
    // Horizontal convolution
    sobel_convolution<<<numBlocksTwoDimConvolution, numThreadTwoDimConvolution, sharedMemory * sizeof(Pixel)>>>(sobelHorizontal, source, 1, width, height);

    sobel_convolution<<<numBlocksTwoDimConvolution, numThreadTwoDimConvolution, sharedMemory * sizeof(Pixel)>>>(sobelVertical, source, 2, width, height);

    harris_matrix<<<numBlockLinear, numThreadLinear>>>(sobelHorizontal, sobelVertical, sobelHorizontalVertical, width, height);

    harris_matrix_sum<<<numBlocksTwoDimConvolution, numThreadTwoDimConvolution, sharedMemory * sizeof(float)>>>(sobelHorizontalSum, sobelHorizontal, width, height);
    harris_matrix_sum<<<numBlocksTwoDimConvolution, numThreadTwoDimConvolution, sharedMemory * sizeof(float)>>>(sobelVerticalSum, sobelVertical, width, height);
    harris_matrix_sum<<<numBlocksTwoDimConvolution, numThreadTwoDimConvolution, sharedMemory * sizeof(float)>>>(sobelHorizontalVerticalSum, sobelHorizontalVertical, width, height);

    harris_final_combination<<<numBlockLinear, numThreadLinear>>>(destination, sobelHorizontalSum, sobelVerticalSum, sobelHorizontalVerticalSum, width, height);

    cudaDeviceSynchronize();
    time = Utilities::seconds() - time;

    cudaFree(sobelHorizontal);
    cudaFree(sobelVertical);
    cudaFree(sobelHorizontalVertical);
    cudaFree(sobelHorizontalSum);
    cudaFree(sobelVerticalSum);
    cudaFree(sobelHorizontalVerticalSum);
    return time;
}

void CudaInterface::harris(Pixel *destination, float *sobelHorizontal, float *sobelVertical, float *sobelHorizontalVertical,
                           float *sobelHorizontalSum, float *sobelVerticalSum, float *sobelHorizontalVerticalSum, int width, int height,
                           int numBlocksTwoDimConvolution, int numThreadTwoDimConvolution, int numBlockLinear,
                           int numThreadLinear, cudaStream_t& stream) {

    int sharedMemory = ((width * height) / (numBlocksTwoDimConvolution * numThreadTwoDimConvolution) + 1) * numThreadTwoDimConvolution * 3 + 6;

    harris_matrix<<<numBlockLinear, numThreadLinear, 0, stream>>>(sobelHorizontal, sobelVertical, sobelHorizontalVertical, width, height);

    harris_matrix_sum<<<numBlocksTwoDimConvolution, numThreadTwoDimConvolution, sharedMemory * sizeof(float), stream>>>(sobelHorizontalSum, sobelHorizontal, width, height);
    harris_matrix_sum<<<numBlocksTwoDimConvolution, numThreadTwoDimConvolution, sharedMemory * sizeof(float), stream>>>(sobelVerticalSum, sobelVertical, width, height);
    harris_matrix_sum<<<numBlocksTwoDimConvolution, numThreadTwoDimConvolution, sharedMemory * sizeof(float), stream>>>(sobelHorizontalVerticalSum, sobelHorizontalVertical, width, height);

    harris_final_combination<<<numBlockLinear, numThreadLinear, 0, stream>>>(destination, sobelHorizontalSum, sobelVerticalSum, sobelHorizontalVerticalSum, width, height);

}