//
// Created by michele on 28/05/19.
//

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "cuda/cuda_interface.h"
#include "utilities/parameters.h"
#include "utilities/utilities.h"

using namespace ros;

void readFrame(const sensor_msgs::Image::ConstPtr&, Publisher&, Publisher&);

// Memory in gpu
Pixel *imageSourceCanny, *imageSourceCorner, *imageCornerGpu, *imageSourceGpu, *grayScaleGpu, *gaussianImageGpu, *cannyImageGpu, *transposeImage;
float *mask, *edgeGradient, *sobelHorizontal, *sobelVertical, *sobelHorizontalVertical,
        *sobelHorizontalSum, *sobelVerticalSum, *sobelHorizontalVerticalSum, *finalCombination;
int *edgeDirection;
bool alloc = false;

int main(int argc, char **argv) {

    ros::init(argc, argv, "services");
    ROS_INFO_STREAM("Started door recognizer node that uses gpu");

    Parameters::getInstance().getValues();

    NodeHandle node;

    Publisher publisherCanny, publisherHarris;

    if(Parameters::getInstance().showEdgeImage())
        publisherCanny = node.advertise<sensor_msgs::Image>("door_recognizer/edge_image", 10);

    if(Parameters::getInstance().showCornerImage())
        publisherHarris = node.advertise<sensor_msgs::Image>("door_recognizer/corner_image", 10);

    Subscriber subscriber = node.subscribe<sensor_msgs::Image>(Parameters::getInstance().getTopic(), 10,
                                                               boost::bind(readFrame, _1, publisherCanny, publisherHarris));

    mask = Utilities::getGaussianArrayPinned(Parameters::getInstance().getGaussianMaskSize(),
                                             Parameters::getInstance().getGaussianAlpha());

    while (true) {
        spinOnce();
    }

}

void readFrame(const sensor_msgs::Image::ConstPtr& image, Publisher& publisherCanny, Publisher& publisherHarris){
    if(!alloc){
        int imageSize = image->width * image->height;
        cudaMalloc(&imageSourceGpu, imageSize * sizeof(Pixel));
        cudaMalloc(&grayScaleGpu, imageSize * sizeof(Pixel));
        cudaMalloc(&gaussianImageGpu, imageSize * sizeof(Pixel));
        cudaMalloc(&transposeImage, imageSize * sizeof(Pixel));
        cudaMalloc(&cannyImageGpu, imageSize * sizeof(Pixel));
        cudaMalloc(&edgeGradient, imageSize * sizeof(float));
        cudaMalloc(&edgeDirection, imageSize * sizeof(int));
        cudaMalloc(&sobelHorizontal, imageSize * sizeof(float));
        cudaMalloc(&sobelVertical, imageSize * sizeof(float));
        cudaMalloc(&sobelHorizontalVertical, imageSize * sizeof(float));
        cudaMalloc(&sobelHorizontalSum, imageSize * sizeof(float));
        cudaMalloc(&sobelVerticalSum, imageSize * sizeof(float));
        cudaMalloc(&sobelHorizontalVerticalSum, imageSize * sizeof(float));
        cudaMalloc(&finalCombination, imageSize * sizeof(float));
        cudaMallocHost(&imageSourceCorner, image->width * image->height * sizeof(Pixel));
        cudaMalloc(&imageCornerGpu, image->width * image->height * sizeof(Pixel));
        alloc = true;
    }


    sensor_msgs::Image imageCornerFinal(*image);
    imageCornerFinal.height = image->height;
    imageCornerFinal.width = image->width;
    imageCornerFinal.encoding = image->encoding;

    cudaStream_t streamCanny, streamHarris;
    cudaStreamCreateWithFlags(&streamCanny, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&streamHarris, cudaStreamNonBlocking);

    cudaEvent_t cannyEnd;
    cudaEventCreate(&cannyEnd);


    imageSourceCanny = CudaInterface::getPixelArray(image->data.data(), image->width, image->height);

    // Gray scale
    cudaMemcpyAsync(imageSourceGpu, imageSourceCanny, image->width * image->height * sizeof(Pixel), cudaMemcpyHostToDevice, streamCanny);

    CudaInterface::toGrayScale(grayScaleGpu, imageSourceGpu, image->width, image->height,
                               Parameters::getInstance().getLinearKernelNumBlock(),
                               Parameters::getInstance().getLinearKernelNumThread(), streamCanny);

    // Gaussian filter
    CudaInterface::gaussianFilter(gaussianImageGpu, grayScaleGpu, transposeImage, image->width, image->height,
                                  mask, Parameters::getInstance().getGaussianMaskSize(),
                                  Parameters::getInstance().getConvolutionOneDimKernelNumBlock(),
                                  Parameters::getInstance().getConvolutionOneDimKernelNumThread(), streamCanny);

    // Sobel filter
    CudaInterface::sobelFilter(edgeGradient, edgeDirection, gaussianImageGpu, sobelHorizontal, sobelVertical,
                               image->width, image->height, Parameters::getInstance().getConvolutionTwoDimKernelNumBlock(),
                               Parameters::getInstance().getConvolutionTwoDimKernelNumThread(),
                               Parameters::getInstance().getLinearKernelNumBlock(),
                               Parameters::getInstance().getLinearKernelNumThread(),
                               streamCanny);

    // Non maximum suppression
    CudaInterface::nonMaximumSuppression(cannyImageGpu, edgeGradient, edgeDirection, image->width, image->height,
                                         Parameters::getInstance().getLinearKernelNumBlock(),
                                         Parameters::getInstance().getLinearKernelNumThread(), streamCanny);

    cudaMemcpyAsync(imageCornerGpu, cannyImageGpu, image->width * image->height * sizeof(Pixel), cudaMemcpyDeviceToDevice, streamCanny);

    cudaEventRecord(cannyEnd, streamCanny);



    cudaMemcpyAsync(imageSourceCanny, cannyImageGpu, image->width * image->height * sizeof(Pixel), cudaMemcpyDeviceToHost, streamCanny);

    // HARRIS
    cudaStreamWaitEvent(streamHarris, cannyEnd, 0);
    CudaInterface::harris(imageCornerGpu, sobelHorizontal, sobelVertical, sobelHorizontalVertical, sobelHorizontalSum, sobelVerticalSum,
            sobelHorizontalVerticalSum, finalCombination, image->width, image->height, Parameters::getInstance().getConvolutionTwoDimKernelNumBlock(),
                          Parameters::getInstance().getConvolutionTwoDimKernelNumThread(), Parameters::getInstance().getLinearKernelNumBlock(),
                          Parameters::getInstance().getLinearKernelNumThread(), streamHarris);

    cudaMemcpyAsync(imageSourceCorner, imageCornerGpu, image->width * image->height * sizeof(Pixel), cudaMemcpyDeviceToHost, streamHarris);


    cudaStreamSynchronize(streamCanny);

    if(Parameters::getInstance().showEdgeImage()){
        CudaInterface::pixelArrayToCharArray((uint8_t*) image->data.data(), imageSourceCanny, image->width, image->height);
    }

    cudaStreamSynchronize(streamHarris);

    if(Parameters::getInstance().showCornerImage()){
        CudaInterface::pixelArrayToCharArray((uint8_t*) imageCornerFinal.data.data(), imageSourceCorner, image->width, image->height);
    }

    if(Parameters::getInstance().showEdgeImage())
        publisherCanny.publish(image);

    if(Parameters::getInstance().showCornerImage())
        publisherHarris.publish(imageCornerFinal);

    cudaStreamDestroy(streamCanny);
    cudaStreamDestroy(streamHarris);
    cudaEventDestroy(cannyEnd);
    cudaFreeHost(imageSourceCanny);
}
