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
#include "utilities/time_utilities.h"
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

        double time_start_gray_scale = seconds();
        CpuAlgorithms::getInstance().toGrayScale(image->getOpenCVImage().data, image->getWidth(), image->getHeight());
        double time_end_gray_scale = seconds();
        cout << " - convert to gray scale: " << time_end_gray_scale - time_start_gray_scale << "\n";

        delete image;

        //GPU

        image = new Image();
        image->acquireImage();

        cout << "\nAnalyze image from OpenCV:\n"
                " - width: " << image->getWidth() << "\n" <<
                " - height: " << image->getHeight() << "\n" <<
                "Operations in GPU:\n";

        cout << " - convert to gray scale: " << Parameters::getInstance().getToGrayScaleNumBlock() <<
                " blocks, " << Parameters::getInstance().getToGrayScaleNumThread() <<
                " thread per block:\n";

        int sizeImage = image->getOpenCVImage().rows * image->getOpenCVImage().cols * 3;
        unsigned char *sourceGpu;

        // ----------- Convert to gray scale ---------------- //

        unsigned char *destinationGrayScaleGpu;

        cudaMalloc(&sourceGpu, sizeof(unsigned char) * sizeImage);
        cudaMalloc(&destinationGrayScaleGpu, sizeof(unsigned char) * sizeImage);

        cudaMemcpy(sourceGpu, image->getOpenCVImage().data, sizeImage * sizeof(unsigned char), cudaMemcpyHostToDevice);

        double timeToGrayScale = CudaInterface::toGrayScale(destinationGrayScaleGpu, sourceGpu, image->getWidth(),
                image->getHeight(), Parameters::getInstance().getToGrayScaleNumBlock(), Parameters::getInstance().getToGrayScaleNumThread());

        cudaMemcpy(image->getOpenCVImage().data, destinationGrayScaleGpu, sizeImage * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        imwrite("ciao.jpg", image->getOpenCVImage());

        cudaFree(sourceGpu);
        cudaFree(destinationGrayScaleGpu);

        cout << timeToGrayScale << endl;
    }

}
