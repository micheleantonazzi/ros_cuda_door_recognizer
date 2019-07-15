# ROS CUDA door recognizer
This is a ROS package that's able to recognize doors using image processing methods. It's goal is to analyze in real time the frame captured by a camera connected to the robot in order to find a door. The algorithms are implemented to run in CPU and also in GPU, to improve the performance.

**NB:** the code is written in CUDA C, so is necessary a NVIDIA GPU

This package offers three different ros node:

* **test_performance**: this node applies the entire algorithm on a single image or a single frame captured by the camera. It can be used to evaluate the performance about the CPU and GPU algorithms. Moreover it is useful to execute test on your GPU to set at its best the parameters of the GPU algorithms.
* **door_recognizer:** this node analyzes as quickly as possible frames come by camera in order to find a door. It runs in CPU 
* **door_recognizer_gpu:** this node analyzes frames come by camera to find a door. It runs in GPU and the performance should be better than the CPU version

The goal of the two node that analyze as quickly as possible the frame captured by the camera is to hide the low precision of the algorithm and its possible errors. In this way the robot can analyzes a lot of different images, taken in different positions and angles so as to have a lot of feedback in order to find a door.

## Algorithmic approach

In order to detect a door, the program uses techniques of image processing. In this way some filters are applied to images captured by camera. the filters applied are the following:

* **gray scale:** first of all the image is converted in gray scale. The procedure is very simple: the values of every pixel are changed with an average obtained by the old values RGB

## Performance evaluation and profiling

An interesting aspect to consider is the analysis about the GPU algorithm, in order to evaluate the performance achieved by its execution and to collect some metrics to measure the correctness. This metrics are:

* **branch efficiency:** measure the percentage of branches that follow the main execution path. If its value is 100% the max efficiency is achieved, that means all the branches follow the same execution path
* **achieved occupancy:** this metrics is about the number of active warp. Its range of values is between 0 and 1, where 1 represent the maximum and the high efficiency
*   