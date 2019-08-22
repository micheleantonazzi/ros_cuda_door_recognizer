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

    // Number of block used to run a linear kernel
    int linear_kernel_num_block;

    // Number of thread per block used to run a linear kernel
    int linear_kernel_num_thread;

    // Number of blocks used to run a one dimensional convolution kernel
    int convolution_one_dim_kernel_num_block;

    // Number of thread per block used to run a one dimensional convolution kernel
    int convolution_one_dim_kernel_num_thread;

    // Number of blocks used to run a two dimensional convolution kernel
    int convolution_two_dim_kernel_num_block;

    // Number of thread per block used to run a two dimensional convolution kernel
    int convolution_two_dim_kernel_num_thread;

    bool show_edge_image;

    bool show_corner_image;

    // The path where to put the processed images
    string processed_images_path;

    int gaussian_mask_size;

    float gaussian_alpha;

    float heightL;

    float heightH;

    float widthL;

    float widthH;

    float directionL;

    float directionH;

    float parallel;

    float ratioL;

    float ratioH;

    Parameters();

public:

    static Parameters& getInstance();

    // Acquires parameter's values
    void getValues();

    bool usingCamera();

    string getImagePath();

    string getTopic();

    int getLinearKernelNumBlock();

    int getLinearKernelNumThread();

    int getConvolutionOneDimKernelNumBlock();

    int getConvolutionOneDimKernelNumThread();

    int getConvolutionTwoDimKernelNumBlock();

    int getConvolutionTwoDimKernelNumThread();
    
    bool showEdgeImage();
    
    bool showCornerImage();

    string getProcessedImagesPath();

    int getGaussianMaskSize();

    float getGaussianAlpha();

    float getHeightL();

    float getHeightH();

    float getWidthL();

    float getWidthH();

    float getDirectionL();

    float getDirectionH();

    float getParallel();

    float getRatioL();

    float getRatioH();
};


#endif //ROS_CUDA_DOOR_RECOGNIZER_PARAMETERS_H
