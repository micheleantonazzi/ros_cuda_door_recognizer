//
// Created by michele on 28/05/19.
//

#include <ros/ros.h>

using namespace ros;

int main(int argc, char **argv){

    ros::init(argc, argv, "services");
    ROS_INFO_STREAM("Hello ROS");
}
