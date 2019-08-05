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

void readFrame(const sensor_msgs::Image::ConstPtr&, Publisher&, Publisher&);

int main(int argc, char **argv){

    ros::init(argc, argv, "services");
    ROS_INFO_STREAM("Started door recognizer node");

    Parameters::getInstance().getValues();

    NodeHandle node;

    Publisher publisherEdge, publisherCorner;

    if(Parameters::getInstance().showEdgeImage())
        publisherEdge = node.advertise<sensor_msgs::Image>("door_recognizer/edge_image", 10);

    if(Parameters::getInstance().showCornerImage())
        publisherCorner = node.advertise<sensor_msgs::Image>("door_recognizer/corner_image", 10);

    Subscriber subscriber = node.subscribe<sensor_msgs::Image>(Parameters::getInstance().getTopic(), 10,
            boost::bind(readFrame, _1, publisherEdge, publisherCorner));

    spin();

}

void readFrame(const sensor_msgs::Image::ConstPtr& image, Publisher& publisherEdge, Publisher& publisherCorner){

    // Create and set the new image
    sensor_msgs::Image imageCannyFinal;
    imageCannyFinal.height = image->height;
    imageCannyFinal.width = image->width;
    imageCannyFinal.encoding = image->encoding;

    sensor_msgs::Image imageCornerFinal;
    imageCornerFinal.height = image->height;
    imageCornerFinal.width = image->width;
    imageCornerFinal.encoding = image->encoding;

    // Array with pixels
    uint8_t *imageGrayData = new uint8_t[imageCannyFinal.height * imageCannyFinal.width * 3];
    uint8_t *imageGaussianData = new uint8_t[imageCannyFinal.height * imageCannyFinal.width * 3];
    uint8_t *imageCanny = new uint8_t[imageCannyFinal.height * imageCannyFinal.width * 3];
    uint8_t *corner = new uint8_t[imageCannyFinal.height * imageCannyFinal.width * 3];

    // Image to gray scale
    CpuAlgorithms::getInstance().toGrayScale(imageGrayData, image->data.data(), imageCannyFinal.width, imageCannyFinal.height);

    // Gaussian Filter
    float *mask = Utilities::getGaussianMatrix(Parameters::getInstance().getGaussianMaskSize(),
                                               Parameters::getInstance().getGaussianAlpha());

    CpuAlgorithms::getInstance().gaussianFilter(imageGaussianData, imageGrayData, mask, imageCannyFinal.width, imageCannyFinal.height,
                                                Parameters::getInstance().getGaussianMaskSize());
    
    float *edgeGradient = new float[imageCannyFinal.width * imageCannyFinal.height];
    int *edgeDirection = new int[imageCannyFinal.width * imageCannyFinal.height];
    
    CpuAlgorithms::getInstance().sobel(edgeGradient, edgeDirection, imageGaussianData, imageCannyFinal.width, imageCannyFinal.height);

    CpuAlgorithms::getInstance().nonMaximumSuppression(imageCanny, edgeGradient, edgeDirection, imageCannyFinal.width, imageCannyFinal.height);

    if(Parameters::getInstance().showEdgeImage()){
        CpuAlgorithms::getInstance().copyArrayToImage(imageCannyFinal, imageCanny);

        publisherEdge.publish(imageCannyFinal);
    }

    CpuAlgorithms::getInstance().harris(corner, imageGaussianData, imageCanny, imageCannyFinal.width, imageCannyFinal.height);

    if(Parameters::getInstance().showCornerImage()){
        CpuAlgorithms::getInstance().copyArrayToImage(imageCornerFinal, corner);

        publisherCorner.publish(imageCornerFinal);
    }



    delete(imageGrayData);
    delete(imageGaussianData);
    delete(mask);
    delete(edgeDirection);
    delete(edgeGradient);
    delete(imageCanny);
    delete(corner);
}
