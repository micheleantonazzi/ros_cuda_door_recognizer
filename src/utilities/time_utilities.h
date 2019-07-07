//
// Created by michele on 07/07/19.
//

#include <sys/time.h>

#ifndef ROS_CUDA_DOOR_RECOGNIZER_TIME_UTILITIES_H
#define ROS_CUDA_DOOR_RECOGNIZER_TIME_UTILITIES_H

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif //ROS_CUDA_DOOR_RECOGNIZER_TIME_UTILITIES_H
