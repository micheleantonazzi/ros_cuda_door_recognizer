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
    nodeHandle.param<string>("image_path", this->image_path, "/home/michele/catkin_ws/src/ros_cuda_door_recognizer/images/test/door_med_res.jpg");
    nodeHandle.param<string>("topic", this->topic, CAMERA_TOPIC);
    nodeHandle.param<int>("linear_kernel_num_block", this->linear_kernel_num_block, 300);
    nodeHandle.param<int>("linear_kernel_num_thread", this->linear_kernel_num_thread, 256);
    nodeHandle.param<int>("convolution_one_dim_kernel_num_block", this->convolution_one_dim_kernel_num_block, 300);
    nodeHandle.param<int>("convolution_one_dim_kernel_num_thread", this->convolution_one_dim_kernel_num_thread, 256);
    nodeHandle.param<int>("convolution_two_dim_kernel_num_block", this->convolution_two_dim_kernel_num_block, 300);
    nodeHandle.param<int>("convolution_two_dim_kernel_num_thread", this->convolution_two_dim_kernel_num_thread, 256);
    nodeHandle.param<bool>("show_edge_image", this->show_edge_image, true);
    nodeHandle.param<bool>("show_corner_image", this->show_corner_image, true);
    nodeHandle.param<bool>("show_door_image", this->show_door_image, true);
    nodeHandle.param<string>("processed_images_path", this->processed_images_path, "/home/michele/catkin_ws/src/ros_cuda_door_recognizer/images/processed_images/");
    nodeHandle.param<int>("gaussian_mask_size", this->gaussian_mask_size, 5);
    nodeHandle.param<float>("gaussian_alpha", this->gaussian_alpha, 0.84);
    nodeHandle.param<float>("heightL", this->heightL, 0.5);
    nodeHandle.param<float>("heightH", this->heightH, 0.9);
    nodeHandle.param<float>("widthL", this->widthL, 0.1);
    nodeHandle.param<float>("widthH", this->widthH, 0.8);
    nodeHandle.param<float>("directionL", this->directionL, 15);
    nodeHandle.param<float>("directionH", this->directionH, 85);
    nodeHandle.param<float>("parallel", this->parallel, 1.5);
    nodeHandle.param<float>("ratioL", this->ratioL, 2);
    nodeHandle.param<float>("ratioH", this->ratioH, 3);
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

int Parameters::getLinearKernelNumBlock() {
    return this->linear_kernel_num_block;
}

int Parameters::getLinearKernelNumThread() {
    return this->linear_kernel_num_thread;
}

int Parameters::getConvolutionOneDimKernelNumBlock() {
    return this->convolution_one_dim_kernel_num_block;
}

int Parameters::getConvolutionOneDimKernelNumThread() {
    return this->convolution_one_dim_kernel_num_thread;
}

int Parameters::getConvolutionTwoDimKernelNumBlock() {
    return this->convolution_two_dim_kernel_num_block;
}
int Parameters::getConvolutionTwoDimKernelNumThread() {
    return this->convolution_two_dim_kernel_num_thread;
}

bool Parameters::showEdgeImage() {
    return this->show_edge_image;
}

bool Parameters::showCornerImage() {
    return this->show_corner_image;
}

bool Parameters::showDoorImage() {
    return this->show_door_image;
}

string Parameters::getProcessedImagesPath() {
    return this->processed_images_path;
}

int Parameters::getGaussianMaskSize() {
    return this->gaussian_mask_size;
}

float Parameters::getGaussianAlpha() {
    return this->gaussian_alpha;
}

float Parameters::getHeightL() {
    return this->heightL;
}

float Parameters::getHeightH() {
    return this->heightH;
}

float Parameters::getWidthL() {
    return this->widthL;
}

float Parameters::getWidthH() {
    return this->widthH;
}

float Parameters::getDirectionL() {
    return this->directionL;
}

float Parameters::getDirectionH() {
    return this->directionH;
}

float Parameters::getParallel() {
    return this->parallel;
}

float Parameters::getRatioL() {
    return this->ratioL;
}

float Parameters::getRatioH() {
    return this->ratioH;
}
