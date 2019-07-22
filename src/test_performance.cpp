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

        imwrite("cpu-grayscale.jpg", image->getOpenCVImage());

        // ------------ Apply gaussian filter --------------- //

        Mat imageGaussian(image->getOpenCVImage());

        float *gaussianFilter = Utilities::getGaussianMatrix(5, 0.84);

        time = Utilities::seconds();
        CpuAlgorithms::getInstance().gaussianFilter(imageGaussian.data, image->getOpenCVImage().data, gaussianFilter, image->getWidth(),
                image->getHeight(), 5);
        time = Utilities::seconds() - time;

        cout << " - apply gaussian filter: " << time << "\n";

        imwrite("cpu-gaussian.jpg", imageGaussian);

        delete image;
        delete gaussianFilter;

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

        cout << " - convert to gray scale: " << Parameters::getInstance().getToGrayScaleNumBlock() <<
                " blocks, " << Parameters::getInstance().getToGrayScaleNumThread() <<
                " thread per block: ";

        int sizeImage = image->getHeight() * image->getWidth();

        Pixel *imageSource = CudaInterface::getPixelArray(image->getOpenCVImage().data, image->getWidth(), image->getHeight());

        Pixel *imageSourceGpu;
        cudaMalloc(&imageSourceGpu, sizeof(Pixel) * sizeImage);

        // ----------- Convert to gray scale ---------------- //

        Pixel *destinationGrayScaleGpu;

        cudaMalloc(&destinationGrayScaleGpu, sizeof(Pixel) * sizeImage);

        cudaMemcpy(imageSourceGpu, imageSource, sizeImage * sizeof(Pixel), cudaMemcpyHostToDevice);

        double timeToGrayScale = CudaInterface::toGrayScale(destinationGrayScaleGpu, imageSourceGpu, image->getWidth(),
                image->getHeight(), Parameters::getInstance().getToGrayScaleNumBlock(), Parameters::getInstance().getToGrayScaleNumThread());

        cudaMemcpy(imageSource, destinationGrayScaleGpu, sizeImage * sizeof(Pixel), cudaMemcpyDeviceToHost);

        CudaInterface::pixelArrayToCharArray(image->getOpenCVImage().data, imageSource, image->getWidth(), image->getHeight());

        imwrite("gpu-grayscale.jpg", image->getOpenCVImage());

        cout << timeToGrayScale << endl;

        // ----------------- Apply Gaussian filter --------------- //

        Pixel *destinationGaussianFilterGpu;
        cudaMalloc(&destinationGaussianFilterGpu, sizeof(Pixel) * sizeImage);

        float *gaussianArray = Utilities::getGaussianArrayPinned(5, 1.3);

        CudaInterface::gaussianFilter(destinationGaussianFilterGpu, destinationGrayScaleGpu, image->getWidth(), image->getHeight(),
                                      gaussianArray, 5, Parameters::getInstance().getGaussianFilterNumBlock(),
                                      Parameters::getInstance().getGaussianFilterNumThread());

        cudaMemcpy(imageSource, destinationGaussianFilterGpu, sizeImage * sizeof(Pixel), cudaMemcpyDeviceToHost);

        CudaInterface::pixelArrayToCharArray(image->getOpenCVImage().data, imageSource, image->getWidth(), image->getHeight());

        imwrite("gpu-gaussian-filter.jpg", image->getOpenCVImage());

        cudaFreeHost(imageSource);
        cudaFree(imageSourceGpu);
        cudaFree(destinationGrayScaleGpu);
        cudaFree(destinationGaussianFilterGpu);
        cudaFreeHost(gaussianArray);

        /*float *m = Utilities::getGaussianMatrix(5, 0.8);
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                printf("%.20f ", m[i * 5 + j]);
            }
            printf("\n");
        }

        m = Utilities::getGaussianArray(5, 0.8);
        for (int i = 0; i < 5; ++i) {
            printf("%.20f ", m[i]);
        }
        printf("\n");
         */
    }
}
