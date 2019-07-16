//
// Created by michele on 28/05/19.
//

#include "../utilities/pixel24.h"

#ifndef ROS_CUDA_DOOR_RECOGNIZER_CUDA_INTERFACE_H
#define ROS_CUDA_DOOR_RECOGNIZER_CUDA_INTERFACE_H

class CudaInterface{

public:

    static void test_cuda();

    // Build an array of Pixel24 that contains image data.
    // The array is allocated in PINNED MEMORY
    static Pixel24* getPixelArray(unsigned char *imageData, int width, int height);

    static void pixelArrayToCharArray(unsigned char *imageData, Pixel24 *source, int width, int height);

    // Convert an image in gray scale
    // Return the time to execute del kernel
    static double toGrayScale(unsigned char *destination, unsigned char *source, int width, int height, int numBlocks, int numThread);

    // Convert an image in gray scale using type Pixel32 to improve the access memory performance
    static double toGrayScale(Pixel24 *destination, Pixel24 *source, int width, int height, int numBlocks, int numThread);
};


#endif //ROS_CUDA_DOOR_RECOGNIZER_CUDA_INTERFACE_H
