//
// Created by michele on 28/05/19.
//

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "cuda/cuda_interface.h"
#include "utilities/parameters.h"

using namespace ros;

void readFrame(const sensor_msgs::Image::ConstPtr&, Publisher&);

int main(int argc, char **argv){

    ros::init(argc, argv, "services");
    ROS_INFO_STREAM("Started door recognizer node that uses gpu");

    Parameters::getInstance().getValues();

    NodeHandle node;

    Publisher publisherGrayScale = node.advertise<sensor_msgs::Image>("door_recognizer/gray_scale", 10);

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

    Pixel *imageSourceGpu, *grayScaleGpu;

    cudaMalloc(&imageSourceGpu, imageSize * sizeof(Pixel));
    cudaMalloc(&grayScaleGpu, imageSize * sizeof(Pixel));

    Pixel *imageSource = CudaInterface::getPixelArray(image->data.data(), image->width, image->height);
    cudaMemcpyAsync(imageSourceGpu, imageSource, imageSize * sizeof(Pixel), cudaMemcpyHostToDevice, stream);

    CudaInterface::toGrayScale(grayScaleGpu, imageSourceGpu, image->width, image->height,
            Parameters::getInstance().getToGrayScaleNumBlock(), Parameters::getInstance().getToGrayScaleNumThread(), stream);

    cudaMemcpyAsync(imageSource, grayScaleGpu, imageSize * sizeof(Pixel), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    CudaInterface::pixelArrayToCharArray((uint8_t*)image->data.data(), imageSource, image->width, image->height);
    publisherGrayScale.publish(image);

    cudaFreeHost(imageSource);
    cudaFree(imageSourceGpu);
    cudaFree(grayScaleGpu);
}
