//
// Created by michele on 07/07/19.
//

#include <ros/ros.h>
#include "parameters.h"

using namespace ros;

string Parameters::DOOR_HIGH_RES = "/home/michele/catkin_ws/src/ros_cuda_door_recognizer/images/test/door_high_res.jpg";
string Parameters::DOOR_MED_RES = "/home/michele/catkin_ws/src/ros_cuda_door_recognizer/images/test/door_med_res.jpg";
string Parameters::CAMERA_TOPIC = "/usb_cam/image_raw";

Parameters& Parameters::getInstance() {
    static Parameters parameters;
    return parameters;
}

Parameters::Parameters() : camera(false), topic(), image_path(){}

void Parameters::getValues() {
    NodeHandle nodeHandle("~");

    nodeHandle.param<bool>("camera", this->camera, false);
    nodeHandle.param<string>("image", this->image_path, DOOR_HIGH_RES);
    nodeHandle.param<string>("topic", this->topic, CAMERA_TOPIC);


}

bool Parameters::usingCamera() {
    return this->camera;
}

string Parameters::getImagePath() {
    return this->image_path;
}

string Parameters::getTopic(){
    return this->topic;
}
