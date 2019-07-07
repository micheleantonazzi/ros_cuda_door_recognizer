//
// Created by michele on 07/07/19.
//

#include <opencv2/opencv.hpp>

using namespace cv;

#ifndef ROS_CUDA_DOOR_RECOGNIZER_IMAGE_H
#define ROS_CUDA_DOOR_RECOGNIZER_IMAGE_H


class Image {
private:

    int image_acquire;

    int width;
    int height;

    Mat image_from_opencv;

public:

    static const int IMAGE_ACQUIRED_BY_CAMERA = 1;
    static const int IMAGE_NOT_ACQUIRED = 0;
    static const int IMAGE_ACQUIRED_BY_OPENCV = -1;

    // Return true if the image is acquired by camera, false otherwise
    bool acquireImage();
    Mat getOpencvImage();
};


#endif //ROS_CUDA_DOOR_RECOGNIZER_IMAGE_H
