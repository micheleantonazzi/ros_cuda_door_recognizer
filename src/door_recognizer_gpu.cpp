//
// Created by michele on 28/05/19.
//

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "cuda/cuda_interface.h"
#include "utilities/parameters.h"
#include "utilities/utilities.h"
#include "cpu/cpu_algorithms.h"

using namespace ros;

void readFrame(const sensor_msgs::Image::ConstPtr&, Publisher&, Publisher&, Publisher&);

// Memory in gpu
Pixel *imageSourceCanny, *imageSourceCorner, *imageCornerGpu, *imageSourceGpu, *grayScaleGpu, *gaussianImageGpu, *cannyImageGpu, *transposeImage;
float *mask, *edgeGradient, *sobelHorizontal, *sobelVertical, *sobelHorizontalVertical,
        *sobelHorizontalSum, *sobelVerticalSum, *sobelHorizontalVerticalSum, *finalCombination;

unsigned char *cornerImage, *edgeImage;
int *edgeDirection;
bool alloc = false;

int main(int argc, char **argv) {

    ros::init(argc, argv, "services");
    ROS_INFO_STREAM("Started door recognizer node that uses gpu");

    Parameters::getInstance().getValues();

    NodeHandle node;

    Publisher publisherCanny, publisherHarris, publisherDoorFound;

    if(Parameters::getInstance().showEdgeImage())
        publisherCanny = node.advertise<sensor_msgs::Image>("door_recognizer/edge_image", 1);

    if(Parameters::getInstance().showCornerImage())
        publisherHarris = node.advertise<sensor_msgs::Image>("door_recognizer/corner_image", 1);

    if(Parameters::getInstance().showDoorImage())
        publisherDoorFound = node.advertise<sensor_msgs::Image>("door_recognizer/door_found", 1);

    Subscriber subscriber = node.subscribe<sensor_msgs::Image>(Parameters::getInstance().getTopic(), 1,
                                                               boost::bind(readFrame, _1, publisherCanny, publisherHarris, publisherDoorFound));

    mask = Utilities::getGaussianArrayPinned(Parameters::getInstance().getGaussianMaskSize(),
                                             Parameters::getInstance().getGaussianAlpha());

    while (true) {
        spinOnce();
    }

}

void readFrame(const sensor_msgs::Image::ConstPtr& image, Publisher& publisherCanny, Publisher& publisherHarris, Publisher& publisherDoorFound){
    if(!alloc){
        int imageSize = image->width * image->height;
        cudaMalloc(&imageSourceGpu, imageSize * sizeof(Pixel));
        cudaMalloc(&grayScaleGpu, imageSize * sizeof(Pixel));
        cudaMalloc(&gaussianImageGpu, imageSize * sizeof(Pixel));
        cudaMalloc(&transposeImage, imageSize * sizeof(Pixel));
        cudaMalloc(&cannyImageGpu, imageSize * sizeof(Pixel));
        cudaMalloc(&edgeGradient, imageSize * sizeof(float));
        cudaMalloc(&edgeDirection, imageSize * sizeof(int));
        cudaMalloc(&sobelHorizontal, imageSize * sizeof(float));
        cudaMalloc(&sobelVertical, imageSize * sizeof(float));
        cudaMalloc(&sobelHorizontalVertical, imageSize * sizeof(float));
        cudaMalloc(&sobelHorizontalSum, imageSize * sizeof(float));
        cudaMalloc(&sobelVerticalSum, imageSize * sizeof(float));
        cudaMalloc(&sobelHorizontalVerticalSum, imageSize * sizeof(float));
        cudaMalloc(&finalCombination, imageSize * sizeof(float));
        cudaMallocHost(&imageSourceCorner, image->width * image->height * sizeof(Pixel));
        cudaMalloc(&imageCornerGpu, image->width * image->height * sizeof(Pixel));
        cornerImage = new unsigned char[image->width * image->height * 3];
        edgeImage = new unsigned char[image->width * image->height * 3];
        alloc = true;
    }


    sensor_msgs::Image imageCornerFinal(*image);
    imageCornerFinal.height = image->height;
    imageCornerFinal.width = image->width;
    imageCornerFinal.encoding = image->encoding;

    sensor_msgs::Image imageDoorFound(*image);
    imageDoorFound.height = image->height;
    imageDoorFound.width = image->width;
    imageDoorFound.encoding = image->encoding;

    cudaStream_t streamCanny, streamHarris;
    cudaStreamCreateWithFlags(&streamCanny, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&streamHarris, cudaStreamNonBlocking);

    cudaEvent_t cannyEnd;
    cudaEventCreate(&cannyEnd);


    imageSourceCanny = CudaInterface::getPixelArray(image->data.data(), image->width, image->height);

    // Gray scale
    cudaMemcpyAsync(imageSourceGpu, imageSourceCanny, image->width * image->height * sizeof(Pixel), cudaMemcpyHostToDevice, streamCanny);

    CudaInterface::toGrayScale(grayScaleGpu, imageSourceGpu, image->width, image->height,
                               Parameters::getInstance().getLinearKernelNumBlock(),
                               Parameters::getInstance().getLinearKernelNumThread(), streamCanny);

    // Gaussian filter
    CudaInterface::gaussianFilter(gaussianImageGpu, grayScaleGpu, transposeImage, image->width, image->height,
                                  mask, Parameters::getInstance().getGaussianMaskSize(),
                                  Parameters::getInstance().getConvolutionOneDimKernelNumBlock(),
                                  Parameters::getInstance().getConvolutionOneDimKernelNumThread(), streamCanny);

    // Sobel filter
    CudaInterface::sobelFilter(edgeGradient, edgeDirection, gaussianImageGpu, sobelHorizontal, sobelVertical,
                               image->width, image->height, Parameters::getInstance().getConvolutionTwoDimKernelNumBlock(),
                               Parameters::getInstance().getConvolutionTwoDimKernelNumThread(),
                               Parameters::getInstance().getLinearKernelNumBlock(),
                               Parameters::getInstance().getLinearKernelNumThread(),
                               streamCanny);

    // Non maximum suppression
    CudaInterface::nonMaximumSuppression(cannyImageGpu, edgeGradient, edgeDirection, image->width, image->height,
                                         Parameters::getInstance().getLinearKernelNumBlock(),
                                         Parameters::getInstance().getLinearKernelNumThread(), streamCanny);

    cudaMemcpyAsync(imageCornerGpu, cannyImageGpu, image->width * image->height * sizeof(Pixel), cudaMemcpyDeviceToDevice, streamCanny);

    cudaEventRecord(cannyEnd, streamCanny);



    cudaMemcpyAsync(imageSourceCanny, cannyImageGpu, image->width * image->height * sizeof(Pixel), cudaMemcpyDeviceToHost, streamCanny);

    // HARRIS
    cudaStreamWaitEvent(streamHarris, cannyEnd, 0);
    CudaInterface::harris(imageCornerGpu, sobelHorizontal, sobelVertical, sobelHorizontalVertical, sobelHorizontalSum, sobelVerticalSum,
            sobelHorizontalVerticalSum, finalCombination, image->width, image->height, Parameters::getInstance().getConvolutionTwoDimKernelNumBlock(),
                          Parameters::getInstance().getConvolutionTwoDimKernelNumThread(), Parameters::getInstance().getLinearKernelNumBlock(),
                          Parameters::getInstance().getLinearKernelNumThread(), streamHarris);

    cudaMemcpyAsync(imageSourceCorner, imageCornerGpu, image->width * image->height * sizeof(Pixel), cudaMemcpyDeviceToHost, streamHarris);

    // CANNY END
    cudaStreamSynchronize(streamCanny);
    CudaInterface::pixelArrayToCharArray(edgeImage, imageSourceCanny, image->width, image->height);

    if(Parameters::getInstance().showEdgeImage()){
        CudaInterface::pixelArrayToCharArray((uint8_t*) image->data.data(), imageSourceCanny, image->width, image->height);
    }

    // Find Hough lines and their intersection points
    vector<Point> intersectionPoints;
    Mat sobelGray(image->height, image->width, CV_8UC1);
    for (int i = 0; i < image->width * image->height; ++i) {
        sobelGray.data[i] = imageSourceCanny[i];
    }

    CpuAlgorithms::getInstance().houghLinesIntersection(intersectionPoints, sobelGray);

    // HARRIS END
    cudaStreamSynchronize(streamHarris);

    if(Parameters::getInstance().showCornerImage()){
        CudaInterface::pixelArrayToCharArray((uint8_t*) imageCornerFinal.data.data(), imageSourceCorner, image->width, image->height);
    }

    // Find candidate corners, only those near the hough lines intersection
    vector<Point> candidateCorners;
    CudaInterface::pixelArrayToCharArray(cornerImage, imageSourceCorner, image->width, image->height);
    CpuAlgorithms::getInstance().findCandidateCorner(candidateCorners, cornerImage, intersectionPoints, image->width, image->height);

    // Find candidate groups composed by four corners
    vector<pair<vector<Point>, Mat*>> candidateGroups;
    CpuAlgorithms::getInstance().candidateGroups(candidateGroups, candidateCorners, image->width, image->height,
                                                 Parameters::getInstance().getHeightL(), Parameters::getInstance().getHeightH(), Parameters::getInstance().getWidthL(),
                                                 Parameters::getInstance().getWidthH(), Parameters::getInstance().getDirectionL(),
                                                 Parameters::getInstance().getDirectionH(), Parameters::getInstance().getParallel(),
                                                 Parameters::getInstance().getRatioL(), Parameters::getInstance().getRatioH());

    // Match the candidate groups with edges found with Canny filter
    vector<vector<Point>> matchFillRatio;
    CpuAlgorithms::getInstance().fillRatio(matchFillRatio, candidateGroups, edgeImage, image->width, image->height);

    for (int i = 0; i < candidateGroups.size(); ++i) {
        delete candidateGroups[i].second;
    }

    if(matchFillRatio.size() > 0 && Parameters::getInstance().showDoorImage()){
        CpuAlgorithms::getInstance().drawRectangle(imageDoorFound.data.data(), image->width, image->height, matchFillRatio[0][0], matchFillRatio[0][1],
                                                   matchFillRatio[0][2], matchFillRatio[0][3], Scalar(0, 0, 255), 4);
    }

    if(Parameters::getInstance().showEdgeImage())
        publisherCanny.publish(image);

    if(Parameters::getInstance().showCornerImage())
        publisherHarris.publish(imageCornerFinal);

    if(Parameters::getInstance().showDoorImage()){
        publisherDoorFound.publish(imageDoorFound);
    }

    cudaStreamDestroy(streamCanny);
    cudaStreamDestroy(streamHarris);
    cudaEventDestroy(cannyEnd);
    cudaFreeHost(imageSourceCanny);
}
