//
// Created by michele on 07/07/19.
//

#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#ifndef ROS_CUDA_DOOR_RECOGNIZER_PARAMETERS_H
#define ROS_CUDA_DOOR_RECOGNIZER_PARAMETERS_H

// SINGLETON
class Parameters {
private:

    static string DOOR_HIGH_RES;
    static string DOOR_MED_RES;
    static string CAMERA_TOPIC;

    // If you want to analise a frame acquired by camera set the variable true, otherwise false
    bool camera;

    // If you are using the camera specify the image's topic
    string topic;

    // If you don't use the camera specify the image's path
    string image_path;

    // Number of block used to run the kernel that convert an image to gray scale
    int to_gray_scale_num_block;

    // Number of thread per block used to run the kernel that convert an image to gray scale
    int to_gray_scale_num_thread;

    // Number of blocks used to run the kernel that applies the gaussian filter
    int gaussian_filter_num_block;

    // Number of thread per block used to run the kernel that applies the gaussian filter
    int gaussian_filter_num_thread;

    // The path where to put the processed images
    string processed_images_path;

    Parameters();

public:

    static Parameters& getInstance();

    // Acquires parameter's values
    void getValues();

    bool usingCamera();

    string getImagePath();

    string getTopic();

    int getToGrayScaleNumBlock();

    int getToGrayScaleNumThread();

    int getGaussianFilterNumBlock();

    int getGaussianFilterNumThread();

    string getProcessedImagesPath();
};


#endif //ROS_CUDA_DOOR_RECOGNIZER_PARAMETERS_H
