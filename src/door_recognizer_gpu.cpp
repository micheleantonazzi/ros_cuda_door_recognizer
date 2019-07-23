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

int main(int argc, char **argv){

    ros::init(argc, argv, "services");
    ROS_INFO_STREAM("Started door recognizer node that uses gpu");

    Parameters::getInstance().getValues();

    NodeHandle node;

    Publisher publisherGrayScale = node.advertise<sensor_msgs::Image>("door_recognizer/image_processed", 10);

    Subscriber subscriber = node.subscribe<sensor_msgs::Image>(Parameters::getInstance().getTopic(), 10,
                                                               boost::bind(readFrame, _1, publisherGrayScale));

    while (true){
        spinOnce();
    }
}

void readFrame(const sensor_msgs::Image::ConstPtr& image, Publisher& publisherGrayScale){
    int imageSize = image->width * image->height;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    Pixel *imageSourceGpu, *grayScaleGpu, *gaussianImageGpu;

    cudaMalloc(&imageSourceGpu, imageSize * sizeof(Pixel));
    cudaMalloc(&grayScaleGpu, imageSize * sizeof(Pixel));
    cudaMalloc(&gaussianImageGpu, imageSize * sizeof(Pixel));

    Pixel *imageSource = CudaInterface::getPixelArray(image->data.data(), image->width, image->height);

    // Gray scale

    cudaMemcpyAsync(imageSourceGpu, imageSource, imageSize * sizeof(Pixel), cudaMemcpyHostToDevice, stream);

    CudaInterface::toGrayScale(grayScaleGpu, imageSourceGpu, image->width, image->height,
            Parameters::getInstance().getToGrayScaleNumBlock(), Parameters::getInstance().getToGrayScaleNumThread(), stream);

    // Gaussian filter
    float *mask = Utilities::getGaussianArrayPinned(Parameters::getInstance().getGaussianMaskSize(),
            Parameters::getInstance().getGaussianAlpha());

    CudaInterface::gaussianFilter(gaussianImageGpu, grayScaleGpu, image->width, image->height,
                                  mask, Parameters::getInstance().getGaussianMaskSize(), Parameters::getInstance().getGaussianFilterNumBlock(),
                                  Parameters::getInstance().getGaussianFilterNumThread(), stream);


    cudaMemcpyAsync(imageSource, gaussianImageGpu, imageSize * sizeof(Pixel), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    CudaInterface::pixelArrayToCharArray((uint8_t*)image->data.data(), imageSource, image->width, image->height);
    publisherGrayScale.publish(image);

    cudaFreeHost(imageSource);
    cudaFree(imageSourceGpu);
    cudaFree(grayScaleGpu);
    cudaFree(gaussianImageGpu);
    cudaFreeHost(mask);
    cudaStreamDestroy(stream);
}
