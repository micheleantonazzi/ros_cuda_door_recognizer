//
// Created by michele on 07/07/19.
//

#ifndef ROS_CUDA_DOOR_RECOGNIZER_CPU_ALGORITMS_H
#define ROS_CUDA_DOOR_RECOGNIZER_CPU_ALGORITMS_H

// SINGLETON
class CpuAlgorithms {
private:
    CpuAlgorithms();

public:
    static CpuAlgorithms& getInstance();

    void toGrayScale(unsigned char*, int, int);
};


#endif //ROS_CUDA_DOOR_RECOGNIZER_CPU_ALGORITMS_H
