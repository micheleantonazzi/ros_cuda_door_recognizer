//
// Created by michele on 07/07/19.
//

#include <ros/ros.h>
#include "parameters.h"

using namespace ros;

string Parameters::DOOR_HIGH_RES = "/home/michele/catkin_ws/src/ros_cuda_door_recognizer/images/test/door_high_res.jpg";
string Parameters::DOOR_MED_RES = "/home/michele/catkin_ws/src/ros_cuda_door_recognizer/images/test/door_med_res.jpg";

Parameters::Parameters() : camera(false), topic(), image_path(){}

void Parameters::getValues() {
    NodeHandle nodeHandle("~");

    nodeHandle.param<bool>("camera", this->camera, false);
    nodeHandle.param<string>("image", this->image_path, DOOR_MED_RES);
}

bool Parameters::usingCamera() {
    return this->camera;
}

Mat Parameters::getImageFromOpenCV() {
    return this->camera ? Mat() :
           imread(this->image_path);
}