//
// Created by michele on 28/05/19.
//

#ifndef ROS_CUDA_DOOR_RECOGNIZER_CUDA_INTERFACE_H
#define ROS_CUDA_DOOR_RECOGNIZER_CUDA_INTERFACE_H

class CudaInterface{

public:
    static void test_cuda();

    // Return the time to execute del kernel
    static double toGrayScale(unsigned char*, unsigned char*, int, int, int, int);
};


#endif //ROS_CUDA_DOOR_RECOGNIZER_CUDA_INTERFACE_H
