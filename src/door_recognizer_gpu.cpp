//
// Created by michele on 28/05/19.
//

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "cuda/cuda_interface.h"
#include "utilities/parameters.h"
#include "utilities/utilities.h"

using namespace ros;

void readFrame(const sensor_msgs::Image::ConstPtr&, Publisher&);

// Memory in gpu
Pixel *imageSource, *imageSourceGpu, *grayScaleGpu, *gaussianImageGpu, *transposeImage;
float *mask, *edgeGradient, *sobelHorizontal, *sobelVertical, *transposeImage1, *transposeImage2;
int *edgeDirection;
bool alloc = false;

int main(int argc, char **argv){

    ros::init(argc, argv, "services");
    ROS_INFO_STREAM("Started door recognizer node that uses gpu");

    Parameters::getInstance().getValues();

    NodeHandle node;

    Publisher publisherGrayScale = node.advertise<sensor_msgs::Image>("door_recognizer/image_processed", 10);

    Subscriber subscriber = node.subscribe<sensor_msgs::Image>(Parameters::getInstance().getTopic(), 10,
                                                               boost::bind(readFrame, _1, publisherGrayScale));

    mask = Utilities::getGaussianArrayPinned(Parameters::getInstance().getGaussianMaskSize(),
                                                    Parameters::getInstance().getGaussianAlpha());

    while (true){
        spinOnce();
    }

    cudaFreeHost(imageSource);
    cudaFree(imageSourceGpu);
    cudaFree(grayScaleGpu);
    cudaFree(gaussianImageGpu);
    cudaFreeHost(mask);
    cudaFree(transposeImage);
    cudaFree(edgeGradient);
    cudaFree(edgeDirection);
    cudaFree(sobelHorizontal);
    cudaFree(sobelVertical);
    cudaFree(transposeImage1);
    cudaFree(transposeImage2);}

void readFrame(const sensor_msgs::Image::ConstPtr& image, Publisher& publisherGrayScale){
    if(!alloc){
        int imageSize = image->width * image->height;
        cudaMalloc(&imageSourceGpu, imageSize * sizeof(Pixel));
        cudaMalloc(&grayScaleGpu, imageSize * sizeof(Pixel));
        cudaMalloc(&gaussianImageGpu, imageSize * sizeof(Pixel));
        cudaMalloc(&transposeImage, imageSize * sizeof(Pixel));
        cudaMalloc(&edgeGradient, imageSize * sizeof(float));
        cudaMalloc(&edgeDirection, imageSize * sizeof(int));
        cudaMalloc(&sobelHorizontal, imageSize * sizeof(float));
        cudaMalloc(&sobelVertical, imageSize * sizeof(float));
        cudaMalloc(&transposeImage1, imageSize * sizeof(float));
        cudaMalloc(&transposeImage2, imageSize * sizeof(float));
        alloc = true;
    }

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    imageSource = CudaInterface::getPixelArray(image->data.data(), image->width, image->height);

    // Gray scale
    cudaMemcpyAsync(imageSourceGpu, imageSource,  image->width * image->height * sizeof(Pixel), cudaMemcpyHostToDevice, stream);

    CudaInterface::toGrayScale(grayScaleGpu, imageSourceGpu, image->width, image->height,
                               Parameters::getInstance().getLinearKernelNumBlock(),
                               Parameters::getInstance().getLinearKernelNumThread(), stream);

    // Gaussian filter
    CudaInterface::gaussianFilter(gaussianImageGpu, grayScaleGpu, transposeImage, image->width, image->height,
                                  mask, Parameters::getInstance().getGaussianMaskSize(),
                                  Parameters::getInstance().getConvolutionKernelNumBlock(),
                                  Parameters::getInstance().getConvolutionKernelNumThread(), stream);

    // Sobel filter
    CudaInterface::sobelFilter(edgeGradient, edgeDirection, gaussianImageGpu, sobelHorizontal, sobelVertical,
                               transposeImage1, transposeImage2, image->width, image->height,
                               Parameters::getInstance().getConvolutionKernelNumBlock(),
                               Parameters::getInstance().getConvolutionKernelNumThread(),
                               Parameters::getInstance().getLinearKernelNumBlock(),
                               Parameters::getInstance().getLinearKernelNumThread(),
                               stream);

    // Non maximum suppression
    CudaInterface::nonMaximumSuppression(gaussianImageGpu, edgeGradient, edgeDirection, image->width, image->height,
                                         Parameters::getInstance().getLinearKernelNumBlock(),
                                         Parameters::getInstance().getLinearKernelNumThread(), stream);


    cudaMemcpyAsync(imageSource, gaussianImageGpu,  image->width * image->height * sizeof(Pixel), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    CudaInterface::pixelArrayToCharArray((uint8_t*)image->data.data(), imageSource, image->width, image->height);
    publisherGrayScale.publish(image);

    cudaStreamDestroy(stream);
    cudaFreeHost(imageSource);
}
