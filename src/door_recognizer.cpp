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

void readFrame(const sensor_msgs::Image::ConstPtr&, Publisher&, Publisher&, Publisher&);

int main(int argc, char **argv){

    ros::init(argc, argv, "services");
    ROS_INFO_STREAM("Started door recognizer node");

    Parameters::getInstance().getValues();

    NodeHandle node;

    Publisher publisherEdge, publisherCorner, publisherDoorFound;

    if(Parameters::getInstance().showEdgeImage())
        publisherEdge = node.advertise<sensor_msgs::Image>("door_recognizer/edge_image", 1);

    if(Parameters::getInstance().showCornerImage())
        publisherCorner = node.advertise<sensor_msgs::Image>("door_recognizer/corner_image", 1);

    if(Parameters::getInstance().showDoorImage())
        publisherDoorFound = node.advertise<sensor_msgs::Image>("door_recognizer/door_found", 1);

    Subscriber subscriber = node.subscribe<sensor_msgs::Image>(Parameters::getInstance().getTopic(), 1,
            boost::bind(readFrame, _1, publisherEdge, publisherCorner, publisherDoorFound));

    spin();

}

void readFrame(const sensor_msgs::Image::ConstPtr& image, Publisher& publisherEdge, Publisher& publisherCorner, Publisher& publisherDoorFound){

    // Array with pixels
    uint8_t *imageGrayData = new uint8_t[image->height * image->width * 3];
    uint8_t *imageGaussianData = new uint8_t[image->height * image->width * 3];
    uint8_t *imageCanny = new uint8_t[image->height * image->width * 3];
    uint8_t *corner = new uint8_t[image->height * image->width * 3];

    // Image to gray scale
    CpuAlgorithms::getInstance().toGrayScale(imageGrayData, image->data.data(), image->width, image->height);

    // Gaussian Filter
    float *mask = Utilities::getGaussianMatrix(Parameters::getInstance().getGaussianMaskSize(),
                                               Parameters::getInstance().getGaussianAlpha());

    CpuAlgorithms::getInstance().gaussianFilter(imageGaussianData, imageGrayData, mask, image->width, image->height,
                                                Parameters::getInstance().getGaussianMaskSize());
    
    float *edgeGradient = new float[image->width * image->height];
    int *edgeDirection = new int[image->width * image->height];
    
    CpuAlgorithms::getInstance().sobel(edgeGradient, edgeDirection, imageGaussianData, image->width, image->height);

    CpuAlgorithms::getInstance().nonMaximumSuppression(imageCanny, edgeGradient, edgeDirection, image->width, image->height);

    // Harris corner detector
    CpuAlgorithms::getInstance().harris(corner, imageGaussianData, imageCanny, image->width, image->height);

    // Find Hough lines and their intersection points
    vector<Point> intersectionPoints;
    Mat sobelGray(image->height, image->width, CV_8UC1);
    for (int i = 0; i < image->width * image->height; ++i) {
        sobelGray.data[i] = imageCanny[i * 3];
    }
    CpuAlgorithms::getInstance().houghLinesIntersection(intersectionPoints, sobelGray);

    // Find candidate corners, only those near the hough lines intersection
    vector<Point> candidateCorners;
    CpuAlgorithms::getInstance().findCandidateCorner(candidateCorners, corner, intersectionPoints, image->width, image->height);

    // Find candidate groups composed by four corners
    vector<pair<vector<Point>, Mat*>> candidateGroups;
    CpuAlgorithms::getInstance().candidateGroups(candidateGroups, candidateCorners, image->width, image->height,
                                                        Parameters::getInstance().getHeightL(), Parameters::getInstance().getHeightH(), Parameters::getInstance().getWidthL(),
                                                        Parameters::getInstance().getWidthH(), Parameters::getInstance().getDirectionL(),
                                                        Parameters::getInstance().getDirectionH(), Parameters::getInstance().getParallel(),
                                                        Parameters::getInstance().getRatioL(), Parameters::getInstance().getRatioH());

    // Match the candidate groups with edges found with Canny filter
    vector<vector<Point>> matchFillRatio;
    CpuAlgorithms::getInstance().fillRatio(matchFillRatio, candidateGroups, imageCanny, image->width, image->height);

    if(matchFillRatio.size() > 1 && Parameters::getInstance().showDoorImage()){
        CpuAlgorithms::getInstance().drawRectangle(image->data.data(), image->width, image->height, matchFillRatio[1][0], matchFillRatio[1][1],
                                                   matchFillRatio[1][2], matchFillRatio[1][3], Scalar(0, 0, 255), 4);
    }


    if(Parameters::getInstance().showEdgeImage()){
        sensor_msgs::Image imageCannyFinal;
        imageCannyFinal.height = image->height;
        imageCannyFinal.width = image->width;
        imageCannyFinal.encoding = image->encoding;

        CpuAlgorithms::getInstance().copyArrayToImage(imageCannyFinal, imageCanny);

        publisherEdge.publish(imageCannyFinal);
    }

    if(Parameters::getInstance().showCornerImage()){
        sensor_msgs::Image imageCornerFinal;
        imageCornerFinal.height = image->height;
        imageCornerFinal.width = image->width;
        imageCornerFinal.encoding = image->encoding;

        CpuAlgorithms::getInstance().copyArrayToImage(imageCornerFinal, corner);

        publisherCorner.publish(imageCornerFinal);
    }

    if(Parameters::getInstance().showDoorImage()){
        sensor_msgs::Image imageDoorFound;
        imageDoorFound.height = image->height;
        imageDoorFound.width = image->width;
        imageDoorFound.encoding = image->encoding;

        CpuAlgorithms::getInstance().copyArrayToImage(imageDoorFound, (uint8_t *) image->data.data());
        publisherDoorFound.publish(imageDoorFound);
    }




    delete(imageGrayData);
    delete(imageGaussianData);
    delete(mask);
    delete(edgeDirection);
    delete(edgeGradient);
    delete(imageCanny);
    delete(corner);
}
