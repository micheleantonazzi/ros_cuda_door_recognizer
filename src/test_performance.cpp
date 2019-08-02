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
        Mat corner(imageSobel);

        time = Utilities::seconds();
        CpuAlgorithms::getInstance().corner(corner.data, imageGaussian.data, imageSobel.data, image->getWidth(), image->getHeight());
        time = Utilities::seconds() - time;

        cout << " - Harris corner detection: " << time << "\n";

        imwrite(Parameters::getInstance().getProcessedImagesPath() + "cpu-corner.jpg", corner);

        /*Image src;
        src.acquireImage();
        Mat dst;
        Canny(src.getOpenCVImage(), dst, 50, 200, 3);
        vector<Vec2f> lines;
        time = Utilities::seconds();
        HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );
        printf("tempo hough: %f\n", Utilities::seconds() - time);
        Mat dstHough;
        cvtColor(dst, dstHough, COLOR_GRAY2BGR);
        for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            line( dstHough, pt1, pt2, Scalar(0,0,255), 3);
            printf("retta a rho: %f, theta: %f\n ", rho, theta);
        }

        imwrite(Parameters::getInstance().getProcessedImagesPath() + "cpu-houghlines.jpg", dstHough);

        */
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
                " - Shared memory per multiprocessor: " << deviceProp.sharedMemPerMultiprocessor / 1024<< " KB\n";

        //GPU

        image = new Image();
        image->acquireImage();

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
             "    - " << Parameters::getInstance().getConvolutionKernelNumBlock() <<
             " blocks, " << Parameters::getInstance().getConvolutionKernelNumThread() <<
             " thread" << endl;

        cudaMemcpy(imageSource, destinationGrayScaleGpu, sizeImage * sizeof(Pixel), cudaMemcpyDeviceToHost);

        CudaInterface::pixelArrayToCharArray(image->getOpenCVImage().data, imageSource, image->getWidth(), image->getHeight());

        imwrite(Parameters::getInstance().getProcessedImagesPath() + "gpu-grayscale.jpg", image->getOpenCVImage());

        // ----------------- Apply Gaussian filter --------------- //
        Pixel *destinationGaussianFilterGpu;
        cudaMalloc(&destinationGaussianFilterGpu, sizeof(Pixel) * sizeImage);

        float *gaussianArray = Utilities::getGaussianArrayPinned(Parameters::getInstance().getGaussianMaskSize(),
                                                                 Parameters::getInstance().getGaussianAlpha());

        time = CudaInterface::gaussianFilter(destinationGaussianFilterGpu, destinationGrayScaleGpu, image->getWidth(), image->getHeight(),
                                      gaussianArray, Parameters::getInstance().getGaussianMaskSize(),
                                             Parameters::getInstance().getConvolutionKernelNumBlock(),
                                             Parameters::getInstance().getConvolutionKernelNumThread());

        cout << " - apply gaussian filter: " << time << endl <<
                "    - " << Parameters::getInstance().getConvolutionKernelNumBlock() <<
                " blocks, " << Parameters::getInstance().getConvolutionKernelNumThread() <<
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
                                          Parameters::getInstance().getConvolutionKernelNumBlock(),
                                          Parameters::getInstance().getConvolutionKernelNumThread(),
                                          Parameters::getInstance().getLinearKernelNumBlock(),
                                          Parameters::getInstance().getLinearKernelNumThread());

        cout << " - apply sobel filter: " << time << "\n" <<
                "    - convolution operation: " << Parameters::getInstance().getConvolutionKernelNumBlock() << " blocks, " <<
                Parameters::getInstance().getConvolutionKernelNumThread() << " thread\n" <<
                "    - linear operation: " << Parameters::getInstance().getLinearKernelNumBlock() << " blocks, " <<
                Parameters::getInstance().getLinearKernelNumThread() << " thread" << endl;

        time = CudaInterface::nonMaximumSuppression(destinationGaussianFilterGpu, edgeGradientGpu, edgeDirectionGpu,
                image->getWidth(), image->getHeight(), Parameters::getInstance().getLinearKernelNumBlock(),
                                                    Parameters::getInstance().getLinearKernelNumThread());

        cout << " - non maximum suppression: " << time << "\n" <<
             "    - convolution operation: " << Parameters::getInstance().getConvolutionKernelNumBlock() << " blocks, " <<
             Parameters::getInstance().getConvolutionKernelNumThread() << " thread\n" <<
             "    - linear operation: " << Parameters::getInstance().getLinearKernelNumBlock() << " blocks, " <<
             Parameters::getInstance().getLinearKernelNumThread() << " thread" << endl;

        cudaMemcpy(imageSource, destinationGaussianFilterGpu, sizeImage * sizeof(Pixel), cudaMemcpyDeviceToHost);
        CudaInterface::pixelArrayToCharArray(image->getOpenCVImage().data, imageSource, image->getWidth(), image->getHeight());
        imwrite(Parameters::getInstance().getProcessedImagesPath() + "gpu-sobel.jpg", image->getOpenCVImage());

        cudaFreeHost(imageSource);
        cudaFree(imageSourceGpu);
        cudaFree(destinationGrayScaleGpu);
        cudaFree(destinationGaussianFilterGpu);
        cudaFreeHost(gaussianArray);
        cudaFree(edgeDirectionGpu);
        cudaFree(edgeGradientGpu);
    }
}
