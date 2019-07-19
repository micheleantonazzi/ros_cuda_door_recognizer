//
// Created by michele on 18/07/19.
//

#ifndef ROS_CUDA_DOOR_RECOGNIZER_UTILITIES_H
#define ROS_CUDA_DOOR_RECOGNIZER_UTILITIES_H


class Utilities {
public:
    static double seconds();

    static float* getGaussianMatrix(int size, float alpha);
};


#endif //ROS_CUDA_DOOR_RECOGNIZER_UTILITIES_H
