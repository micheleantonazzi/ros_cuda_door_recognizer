//
// Created by michele on 07/07/19.
//

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <string>

#include "utilities/parameters.h"
#include "cpu/cpu_algorithms.h"

using namespace ros;
using namespace std;

void readFrame(const sensor_msgs::Image::ConstPtr&, Publisher&);

int main(int argc, char **argv){

    ros::init(argc, argv, "services");
    ROS_INFO_STREAM("Started door recognizer node");

    Parameters::getInstance().getValues();
    cout << Parameters::getInstance().getTopic() << endl;

    NodeHandle node;

    Publisher publisherGrayScale = node.advertise<sensor_msgs::Image>("gray_scale", 10);

    Subscriber subscriber = node.subscribe<sensor_msgs::Image>(Parameters::getInstance().getTopic(), 10,
            boost::bind(readFrame, _1, publisherGrayScale));

    spin();

}

void readFrame(const sensor_msgs::Image::ConstPtr& image, Publisher& publisherGrayScale){
    sensor_msgs::Image imageGray;
    imageGray.height = image->height;
    imageGray.width = image->width;
    imageGray.encoding = image->encoding;
    CpuAlgorithms::getInstance().toGrayScale(imageGray, *image);
    publisherGrayScale.publish(imageGray);
}
