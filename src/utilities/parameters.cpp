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
    nodeHandle.param<string>("image_path", this->image_path, DOOR_MED_RES);
    nodeHandle.param<string>("topic", this->topic, CAMERA_TOPIC);
    nodeHandle.param<int>("to_gray_scale_num_block", this->to_gray_scale_num_block, 128);
    nodeHandle.param<int>("to_gray_scale_num_thread", this->to_gray_scale_num_thread, 512);

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

int Parameters::getToGrayScaleNumBlock() {
    return this->to_gray_scale_num_block;
}

int Parameters::getToGrayScaleNumThread() {
    return this->to_gray_scale_num_thread;
}
