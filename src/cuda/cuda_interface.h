//
// Created by michele on 28/05/19.
//

#include "../utilities/pixel.h"

#ifndef ROS_CUDA_DOOR_RECOGNIZER_CUDA_INTERFACE_H
#define ROS_CUDA_DOOR_RECOGNIZER_CUDA_INTERFACE_H

class CudaInterface{

public:

    static void test_cuda();

    // Build an array of Pixel24 that contains image data.
    // The array is allocated in PINNED MEMORY
    static Pixel* getPixelArray(const unsigned char *imageData, int width, int height);

    static void pixelArrayToCharArray(unsigned char *imageData, Pixel *source, int width, int height);

    // Convert an image in gray scale
    // Return the time to execute del kernel
    static double toGrayScale(unsigned char *destination, unsigned char *source, int width, int height, int numBlocks, int numThread);

    // Convert an image in gray scale using type Pixel32 to improve the access memory performance
    // This kernel is in a stream non NULL
    static void toGrayScale(Pixel *destination, Pixel *source, int width, int height, int numBlocks, int numThread, const cudaStream_t &stream);

    // Convert an image in gray scale using type Pixel32 to improve the access memory performance
    // The kernel is in the default stream
    // This function returns the kernel execution time
    static double toGrayScale(Pixel *destination, Pixel *source, int width, int height, int numBlocks, int numThread);

    // Apply a Gaussian filter to image
    // To improve the performance are applied to the image two one-dimensional convolution,
    // one horizontally and one vertically using the same kernel
    static double gaussianFilter(Pixel *destination, Pixel *source, int width, int height,
            float *gaussianMask, int maskDim, int NumBlocks, int numThread);

    static void gaussianFilter(Pixel *destination, Pixel *source, int width, int height,
                                 float *gaussianMask, int maskDim, int NumBlocks, int numThread, cudaStream_t &stream);

    // Apply sobel filter to image
    // To improve the performance are applied to the image four one-dimensional convolution,
    // two horizontal and two vertically
    static double sobelFilter(float *edgeGradient, int *edgeDirection, Pixel *source, int width, int height,
            int numBlocks, int numThread);

    static double nonMaximumSuppression(Pixel *destination, float *edgeGradient, int *edgeDirection, int width, int height);
};


#endif //ROS_CUDA_DOOR_RECOGNIZER_CUDA_INTERFACE_H
