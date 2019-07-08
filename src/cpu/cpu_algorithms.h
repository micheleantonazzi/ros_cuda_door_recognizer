//
// Created by michele on 07/07/19.
//

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#ifndef ROS_CUDA_DOOR_RECOGNIZER_CPU_ALGORITMS_H
#define ROS_CUDA_DOOR_RECOGNIZER_CPU_ALGORITMS_H

// SINGLETON
class CpuAlgorithms {
private:
    CpuAlgorithms();

public:
    static CpuAlgorithms& getInstance();

    void toGrayScale(unsigned char*, int, int);
    void toGrayScale(sensor_msgs::Image&, const sensor_msgs::Image&);
};


#endif //ROS_CUDA_DOOR_RECOGNIZER_CPU_ALGORITMS_H
