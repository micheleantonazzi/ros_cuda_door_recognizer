//
// Created by michele on 07/07/19.
//

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include "utilities/parameters.h"
#include "utilities/image.h"
#include "cpu/cpu_algorithms.h"
#include "utilities/utilities.h"
#include "cuda/cuda_interface.h"

using namespace ros;
using namespace cv;

int main(int argc, char **argv){

    ros::init(argc, argv, "services");
    ROS_INFO("Started node to test performance");

    Parameters::getInstance().getValues();

    if(Parameters::getInstance().usingCamera()){

    }
    else {

        //CPU

        Image *image = new Image();
        image->acquireImage();

        cout << "Analyze image from OpenCV:\n"
                " - width: " << image->getWidth() << "\n" <<
                " - height: " << image->getHeight() << "\n" <<
                "Operations in CPU:\n";

        double time = Utilities::seconds();
        CpuAlgorithms::getInstance().toGrayScale(image->getOpenCVImage().data, image->getWidth(), image->getHeight());
        time = Utilities::seconds() - time;
        cout << " - convert to gray scale: " << time << "\n";

        imwrite(Parameters::getInstance().getProcessedImagesPath() + "cpu-grayscale.jpg", image->getOpenCVImage());

        // ------------ Apply gaussian filter --------------- //

        Mat imageGaussian(image->getOpenCVImage());

        float *gaussianFilter = Utilities::getGaussianMatrix(Parameters::getInstance().getGaussianMaskSize(),
                                                             Parameters::getInstance().getGaussianAlpha());

        time = Utilities::seconds();
        CpuAlgorithms::getInstance().gaussianFilter(imageGaussian.data, image->getOpenCVImage().data, gaussianFilter, image->getWidth(),
                image->getHeight(), Parameters::getInstance().getGaussianMaskSize());
        time = Utilities::seconds() - time;

        cout << " - apply gaussian filter: " << time << "\n";

        imwrite(Parameters::getInstance().getProcessedImagesPath() + "cpu-gaussian.jpg", imageGaussian);

        // Sobel filter
        Mat imageSobel(image->getOpenCVImage());

        float *edgeGradient = new float[image->getWidth() * image->getHeight()];
        int *edgeDirection = new int[image->getWidth() * image->getHeight()];

        time = Utilities::seconds();
        CpuAlgorithms::getInstance().sobel(edgeGradient, edgeDirection, imageGaussian.data,
                image->getWidth(), image->getHeight());
        time = Utilities::seconds() - time;
        cout << " - apply sobel filter: " << time << "\n";

        time = Utilities::seconds();
        CpuAlgorithms::getInstance().nonMaximumSuppression(imageSobel.data, edgeGradient, edgeDirection,
                image->getWidth(), image->getHeight());
        time = Utilities::seconds() - time;
        cout << " - non maximum suppression: " << time << "\n";

        imwrite(Parameters::getInstance().getProcessedImagesPath() + "cpu-sobel.jpg", imageSobel);

        // Harris corner detector
        Mat corner(image->getHeight(), image->getWidth(), CV_8UC3);
        for (int j = 0; j < image->getWidth() * image->getHeight() * 3; ++j) {
            *(corner.data + j) = image->getOpenCVImage().data[j];

        }

        time = Utilities::seconds();
        CpuAlgorithms::getInstance().harris(corner.data, imageGaussian.data, imageSobel.data, image->getWidth(),
                                            image->getHeight());
        time = Utilities::seconds() - time;

        cout << " - Harris corner detection: " << time << "\n";

        imwrite(Parameters::getInstance().getProcessedImagesPath() + "cpu-corner.jpg", corner);

        // Find corners
        /*
        vector<Point> corners;
        CpuAlgorithms::getInstance().findCorner(corner.data, corners, image->getWidth(), image->getHeight());
        printf("cap %i\n", corners.size());
        vector<int> groups;
        time = CpuAlgorithms::getInstance().candidateGroups(corners, groups, corner, image->getWidth(), image->getHeight());
        printf("gruppi: %i\n", groups.size());
        printf("Tempo gruppi %f\n", time);

        imwrite(Parameters::getInstance().getProcessedImagesPath() + "cpu-corner-lines.jpg", corner);
         */

        // Find intersections between hough lines
        vector<Point> intersectionPoints;
        Mat sobelGray(image->getHeight(), image->getWidth(), CV_8UC1);
        cvtColor(imageSobel, sobelGray, COLOR_BGR2GRAY);
        time = CpuAlgorithms::getInstance().houghLinesIntersection(intersectionPoints, sobelGray);
        printf(" - find hough lines intersections: %f seconds\n", time);

        // Find candidate corners
        vector<Point> candidateCorners;
        time = CpuAlgorithms::getInstance().findCandidateCorner(candidateCorners, corner.data, intersectionPoints, image->getWidth(), image->getHeight());
        printf(" - find candidate corners: %f seconds\n", time);


        // Find candidate groups
        vector<pair<vector<Point>, Mat*>> candidateGroups;
        time = CpuAlgorithms::getInstance().candidateGroups(candidateGroups, candidateCorners, corner, image->getWidth(), image->getHeight());
        printf(" - find candidate groups: %f seconds\n", time);
        imwrite(Parameters::getInstance().getProcessedImagesPath() + "cpu-corner-lines.jpg", corner);

        vector<vector<Point>> matchFillRatio;
        time = CpuAlgorithms::getInstance().fillRatio(matchFillRatio,candidateGroups, imageSobel.data, image->getWidth(), image->getHeight());

        printf(" - fill ratio: %f seconds\n", time);

        if(matchFillRatio.size() > 0){
            line(image->getOpenCVImage(), matchFillRatio[0][0], matchFillRatio[0][1], Scalar(0, 0, 255), 4);

            line(image->getOpenCVImage(), matchFillRatio[0][1], matchFillRatio[0][2], Scalar(0, 0, 255), 4);

            line(image->getOpenCVImage(), matchFillRatio[0][2], matchFillRatio[0][3], Scalar(0, 0, 255), 4);

            line(image->getOpenCVImage(), matchFillRatio[0][3], matchFillRatio[0][0], Scalar(0, 0, 255), 4);
        }


        imwrite(Parameters::getInstance().getProcessedImagesPath() + "cpu-door-found.jpg", image->getOpenCVImage());


        delete image;
        delete gaussianFilter;
        delete edgeDirection;
        delete edgeGradient;

        cout << endl;

        // GPU info
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cout << "GPU properties:\n"
                " - Name: " << deviceProp.name << "\n"
                " - Global memory size: " << deviceProp.totalGlobalMem / 1024 / 1024  << " MB\n"
                " - Multiprocessors: " << deviceProp.multiProcessorCount << "\n"
                " - Registers per multiprocessor: " << deviceProp.regsPerMultiprocessor << "\n"
                " - Shared memory per multiprocessor: " << deviceProp.sharedMemPerMultiprocessor / 1024 << " KB\n";

        //GPU

        image = new Image();
        image->acquireImage();

        Mat imageSobelOpenCV(image->getHeight(), image->getWidth(), CV_8UC3);

        cout << "\nAnalyze image from OpenCV:\n"
                " - width: " << image->getWidth() << "\n" <<
                " - height: " << image->getHeight() << "\n" <<
                "Operations in GPU:\n";

        int sizeImage = image->getHeight() * image->getWidth();

        Pixel *imageSource = CudaInterface::getPixelArray(image->getOpenCVImage().data, image->getWidth(), image->getHeight());

        Pixel *imageSourceGpu;
        cudaMalloc(&imageSourceGpu, sizeof(Pixel) * sizeImage);

        // ----------- Convert to gray scale ---------------- //

        Pixel *destinationGrayScaleGpu;

        cudaMalloc(&destinationGrayScaleGpu, sizeof(Pixel) * sizeImage);

        cudaMemcpy(imageSourceGpu, imageSource, sizeImage * sizeof(Pixel), cudaMemcpyHostToDevice);

        time = CudaInterface::toGrayScale(destinationGrayScaleGpu, imageSourceGpu, image->getWidth(),
                                          image->getHeight(), Parameters::getInstance().getLinearKernelNumBlock(),
                                          Parameters::getInstance().getLinearKernelNumThread());
        cout << " - convert to gray scale: " << time << endl <<
             "    - " << Parameters::getInstance().getConvolutionOneDimKernelNumBlock() <<
             " blocks, " << Parameters::getInstance().getConvolutionOneDimKernelNumThread() <<
             " thread" << endl;

        cudaMemcpy(imageSource, destinationGrayScaleGpu, sizeImage * sizeof(Pixel), cudaMemcpyDeviceToHost);

        CudaInterface::pixelArrayToCharArray(image->getOpenCVImage().data, imageSource, image->getWidth(), image->getHeight());

        imwrite(Parameters::getInstance().getProcessedImagesPath() + "gpu-grayscale.jpg", image->getOpenCVImage());

        // ----------------- Apply Gaussian filter --------------- //
        Pixel *destinationGaussianFilterGpu;
        Pixel *destinationSobelSuppressedGpu;
        Pixel *destinationHarrisCornerGpu;
        cudaMalloc(&destinationGaussianFilterGpu, sizeof(Pixel) * sizeImage);
        cudaMalloc(&destinationSobelSuppressedGpu, sizeof(Pixel) * sizeImage);
        cudaMalloc(&destinationHarrisCornerGpu, sizeof(Pixel) * sizeImage);

        float *gaussianArray = Utilities::getGaussianArrayPinned(Parameters::getInstance().getGaussianMaskSize(),
                                                                 Parameters::getInstance().getGaussianAlpha());

        time = CudaInterface::gaussianFilter(destinationGaussianFilterGpu, destinationGrayScaleGpu, image->getWidth(), image->getHeight(),
                                      gaussianArray, Parameters::getInstance().getGaussianMaskSize(),
                                             Parameters::getInstance().getConvolutionOneDimKernelNumBlock(),
                                             Parameters::getInstance().getConvolutionOneDimKernelNumThread());

        cout << " - apply gaussian filter: " << time << endl <<
             "    - " << Parameters::getInstance().getConvolutionOneDimKernelNumBlock() <<
             " blocks, " << Parameters::getInstance().getConvolutionOneDimKernelNumThread() <<
             " thread" << endl;

        cudaMemcpy(imageSource, destinationGaussianFilterGpu, sizeImage * sizeof(Pixel), cudaMemcpyDeviceToHost);
        CudaInterface::pixelArrayToCharArray(image->getOpenCVImage().data, imageSource, image->getWidth(), image->getHeight());
        imwrite(Parameters::getInstance().getProcessedImagesPath() + "gpu-gaussian-filter.jpg", image->getOpenCVImage());

        // Apply sobel filter
        float *edgeGradientGpu;
        int *edgeDirectionGpu;

        cudaMalloc(&edgeGradientGpu, image->getWidth() * image->getHeight() * sizeof(float));
        cudaMalloc(&edgeDirectionGpu, image->getWidth() * image->getHeight() * sizeof(int));

        time = CudaInterface::sobelFilter(edgeGradientGpu, edgeDirectionGpu, destinationGaussianFilterGpu, image->getWidth(), image->getHeight(),
                                          Parameters::getInstance().getConvolutionTwoDimKernelNumBlock(),
                                          Parameters::getInstance().getConvolutionTwoDimKernelNumThread(),
                                          Parameters::getInstance().getLinearKernelNumBlock(),
                                          Parameters::getInstance().getLinearKernelNumThread());

        cout << " - apply sobel filter: " << time << "\n" <<
             "    - convolution operation: " << Parameters::getInstance().getConvolutionTwoDimKernelNumBlock() << " blocks, " <<
             Parameters::getInstance().getConvolutionTwoDimKernelNumThread() << " thread\n" <<
             "    - linear operation: " << Parameters::getInstance().getLinearKernelNumBlock() << " blocks, " <<
             Parameters::getInstance().getLinearKernelNumThread() << " thread" << endl;

        time = CudaInterface::nonMaximumSuppression(destinationSobelSuppressedGpu, edgeGradientGpu, edgeDirectionGpu,
                image->getWidth(), image->getHeight(), Parameters::getInstance().getLinearKernelNumBlock(),
                                                    Parameters::getInstance().getLinearKernelNumThread());

        cout << " - non maximum suppression: " << time << "\n" <<
             "    - convolution operation: " << Parameters::getInstance().getConvolutionOneDimKernelNumBlock() << " blocks, " <<
                                                                                                                              Parameters::getInstance().getConvolutionOneDimKernelNumThread() << " thread\n" <<
             "    - linear operation: " << Parameters::getInstance().getLinearKernelNumBlock() << " blocks, " <<
             Parameters::getInstance().getLinearKernelNumThread() << " thread" << endl;

        cudaMemcpy(imageSource, destinationSobelSuppressedGpu, sizeImage * sizeof(Pixel), cudaMemcpyDeviceToHost);
        CudaInterface::pixelArrayToCharArray(image->getOpenCVImage().data, imageSource, image->getWidth(), image->getHeight());
        imwrite(Parameters::getInstance().getProcessedImagesPath() + "gpu-sobel.jpg", image->getOpenCVImage());

        for (int k = 0; k < image->getWidth() * image->getHeight() * 3; ++k) {
            imageSobelOpenCV.data[k] = image->getOpenCVImage().data[k];
        }

        // Corner detection
        time = CudaInterface::harris(destinationSobelSuppressedGpu, destinationGaussianFilterGpu, image->getWidth(), image->getHeight(),
                              Parameters::getInstance().getConvolutionTwoDimKernelNumBlock(),
                              Parameters::getInstance().getConvolutionTwoDimKernelNumThread(),
                              Parameters::getInstance().getLinearKernelNumBlock(), Parameters::getInstance().getLinearKernelNumThread());

        cout << " - Harris corner detector: " << time << "\n" <<
             "    - convolution operation: " << Parameters::getInstance().getConvolutionTwoDimKernelNumBlock() << " blocks, " <<
             Parameters::getInstance().getConvolutionTwoDimKernelNumThread() << " thread\n" <<
             "    - linear operation: " << Parameters::getInstance().getLinearKernelNumBlock() << " blocks, " <<
             Parameters::getInstance().getLinearKernelNumThread() << " thread" << endl;

        cudaMemcpy(imageSource, destinationSobelSuppressedGpu, sizeImage * sizeof(Pixel), cudaMemcpyDeviceToHost);
        CudaInterface::pixelArrayToCharArray(image->getOpenCVImage().data, imageSource, image->getWidth(), image->getHeight());

        imwrite(Parameters::getInstance().getProcessedImagesPath() + "gpu-corner.jpg", image->getOpenCVImage());


        intersectionPoints.clear();
        sobelGray.setTo(0);
        cvtColor(imageSobelOpenCV, sobelGray, COLOR_BGR2GRAY);
        time = CpuAlgorithms::getInstance().houghLinesIntersection(intersectionPoints, sobelGray);
        printf(" - find hough lines intersections: %f seconds\n", time);

        // Find candidate corners
        candidateCorners.clear();
        time = CpuAlgorithms::getInstance().findCandidateCorner(candidateCorners, image->getOpenCVImage().data, intersectionPoints, image->getWidth(), image->getHeight());
        printf(" - find candidate corners: %f seconds\n", time);


        // Find candidate groups
        candidateGroups.clear();
        time = CpuAlgorithms::getInstance().candidateGroups(candidateGroups, candidateCorners, corner, image->getWidth(), image->getHeight());
        printf(" - find candidate groups: %f seconds\n", time);
        imwrite(Parameters::getInstance().getProcessedImagesPath() + "cpu-corner-lines.jpg", corner);

        matchFillRatio.clear();
        time = CpuAlgorithms::getInstance().fillRatio(matchFillRatio,candidateGroups, imageSobelOpenCV.data, image->getWidth(), image->getHeight());

        printf(" - fill ratio: %f seconds\n", time);

        if(matchFillRatio.size() > 0){
            line(image->getOpenCVImage(), matchFillRatio[0][0], matchFillRatio[0][1], Scalar(0, 0, 255), 4);

            line(image->getOpenCVImage(), matchFillRatio[0][1], matchFillRatio[0][2], Scalar(0, 0, 255), 4);

            line(image->getOpenCVImage(), matchFillRatio[0][2], matchFillRatio[0][3], Scalar(0, 0, 255), 4);

            line(image->getOpenCVImage(), matchFillRatio[0][0], matchFillRatio[0][3], Scalar(0, 0, 255), 4);
        }


        imwrite(Parameters::getInstance().getProcessedImagesPath() + "gpu-door-found.jpg", image->getOpenCVImage());

        cudaFreeHost(imageSource);
        cudaFree(imageSourceGpu);
        cudaFree(destinationGrayScaleGpu);
        cudaFree(destinationGaussianFilterGpu);
        cudaFreeHost(gaussianArray);
        cudaFree(edgeDirectionGpu);
        cudaFree(edgeGradientGpu);
        cudaFree(destinationSobelSuppressedGpu);
        cudaFree(destinationHarrisCornerGpu);
    }
}
