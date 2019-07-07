# ROS CUDA door recognizer
This is a ROS package that's able to recognize doors. The algorithms are implemented to run in CPU and also in GPU, to improve the performance. It's goal is to analyze in real time the frame captured by a camera connected to the robot in order to find a door. 

**NB:** the code is written in CUDA C, so is necessary a NVIDIA GPU

This package offers three different ros node:

* **test_performance**: this node can be used to evaluate the performance about the CPU and GPU algorithms. Moreover it is useful to execute test on your GPU to set at its best the GPU algorithms.
* **door_recognizer:** this node analyze frames come by camera to find a door. It runs in CPU 
* **door_recognizer_gpu:** this node analyze frames come by camera to find a door. It runs in GPU 