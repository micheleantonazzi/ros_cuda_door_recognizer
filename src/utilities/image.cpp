//
// Created by michele on 07/07/19.
//

#include "image.h"
#include "parameters.h"

bool Image::acquireImage() {
    if(!Parameters::getInstance().usingCamera()){
        this->image_from_opencv = imread(Parameters::getInstance().getImagePath());
        this->width = this->image_from_opencv.cols;
        this->height = this->image_from_opencv.rows;
        this->image_acquire = this->IMAGE_ACQUIRED_BY_OPENCV;
    }
    else{
        this->image_acquire = this->IMAGE_ACQUIRED_BY_CAMERA;
    }
    return this->image_acquire;
}

int Image::getWidth() {
    return this->width;
}

int Image::getHeight() {
    return this->height;
}

Mat Image::getOpenCVImage() {
    if(this->image_acquire != this->IMAGE_NOT_ACQUIRED && this->image_acquire == this->IMAGE_ACQUIRED_BY_OPENCV)
        return this->image_from_opencv;
    else
        return Mat();
}
