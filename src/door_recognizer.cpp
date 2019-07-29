//
// Created by michele on 07/07/19.
//

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <string>

#include "utilities/parameters.h"
#include "cpu/cpu_algorithms.h"
#include "utilities/utilities.h"

using namespace ros;
using namespace std;

void readFrame(const sensor_msgs::Image::ConstPtr&, Publisher&);

int main(int argc, char **argv){

    ros::init(argc, argv, "services");
    ROS_INFO_STREAM("Started door recognizer node");

    Parameters::getInstance().getValues();

    NodeHandle node;

    Publisher publisherGrayScale = node.advertise<sensor_msgs::Image>("door_recognizer/image_processed", 10);

    Subscriber subscriber = node.subscribe<sensor_msgs::Image>(Parameters::getInstance().getTopic(), 10,
            boost::bind(readFrame, _1, publisherGrayScale));

    spin();

}

void readFrame(const sensor_msgs::Image::ConstPtr& image, Publisher& publisherGrayScale){

    // Create and set the new image
    sensor_msgs::Image imageFinal;
    imageFinal.height = image->height;
    imageFinal.width = image->width;
    imageFinal.encoding = image->encoding;

    // Array with pixels
    uint8_t *imageGrayData = new uint8_t[imageFinal.height * imageFinal.width * 3];
    uint8_t *imageGaussianData = new uint8_t[imageFinal.height * imageFinal.width * 3];

    // Image to gray scale
    CpuAlgorithms::getInstance().toGrayScale(imageGrayData, image->data.data(), imageFinal.width, imageFinal.height);

    // Gaussian Filter
    float *mask = Utilities::getGaussianMatrix(Parameters::getInstance().getGaussianMaskSize(),
                                               Parameters::getInstance().getGaussianAlpha());

    CpuAlgorithms::getInstance().gaussianFilter(imageGaussianData, imageGrayData, mask, imageFinal.width, imageFinal.height,
            Parameters::getInstance().getGaussianMaskSize());
    
    float *edgeGradient = new float[imageFinal.width * imageFinal.height];
    int *edgeDirection = new int[imageFinal.width * imageFinal.height];
    
    CpuAlgorithms::getInstance().sobel(edgeGradient, edgeDirection, imageGaussianData, imageFinal.width, imageFinal.height);

    CpuAlgorithms::getInstance().nonMaximumSuppression(imageGaussianData, edgeGradient, edgeDirection, imageFinal.width, imageFinal.height);
    
    // Copy data to sensor_msgs::Image
    CpuAlgorithms::getInstance().copyArrayToImage(imageFinal, imageGaussianData);

    publisherGrayScale.publish(imageFinal);

    delete(imageGrayData);
    delete(imageGaussianData);
    delete(mask);
    delete(edgeDirection);
    delete(edgeGradient);
}
