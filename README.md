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

* **Gaussian filter:** this filter is important because it prepares the image to the future manipulations. In particular this is a low-pass filter so removes high-frequency components from the image, in order to reduce the image noise. To smooth the image the Gaussian filter is applied by convolution to the image. The kernel (also called mask) of the Gaussian filter is a matrix. Its equation is:

  ![Gaussian kernel 2D](images/md/gaussian_matrix.png)  where *x*, *y* are the distance to origin.

  The problem with a 2D kernel is the high complexity, equal to O(m1 * m2 * n * n), where *m1* is the image width, *m2* is the image height and *n* is the kernel dimension. In order to reduce the complexity, a two-dimensional Gaussian filter can be applied with two single-dimensional filter, using a 1D kernel. This kernel must be applied in horizontal and vertical direction and the effects are the same as those obtained with 2D kernel. Its equation is 

  ![Gaussian kernel 2D](images/md/gaussian_array.png) 
  
  Now the complexity is O(m1 * m2 * n)

## Performance evaluation and profiling

In this section the algorithms implemented in CPU and GPU are compared in order to measure the speedup achieved by the parallel implementation executed in GPU. Another interesting aspect to consider is the profiling of kernels, in order to in order to collect some metrics and measure their correctness. This metrics are:

* **branch efficiency:** measures the percentage of branches that follow the main execution path. If its value is 100% the max efficiency is achieved, that means all the branches follow the same execution path
* **achieved occupancy:** this metrics is about the number of active warp. Its range of values is between 0 and 1, where 1 represent the maximum and the high efficiency
*   **global memory load efficiency:** measures the efficiency with which the kernel read data from global memory. In GPU with memory access it is possible to read 128 byte, in this way if all 32 threads in a warp require a float (4 byte), the data will be given to all threads with a single transaction. But there is some conditions for having this property: the 128 bytes must be consecutive and aligned, so the first address of the transaction must be a multiple of 128. If the threads request for data in arbitrary positions, many transaction are made (each one read 128 byte) but many read values are discarded because they are not requested by the threads. This metric is the percentage of the byte read from the memory and the byte used by the threads, so if the value is 100% the maximum efficiency is achieved
*   **global memory load efficiency:** the principle with which data is store in global memory is the same as written above. So if the value is 100% the pattern is respected and the efficiency is maximum

All the test are executed in a laptop with:

* CPU: Intel Core i5-8250U, 1.60 Ghz, 4 core, 8 thread
* GPU: Nvidia MX 150, 4 GB of GDDR5 memory

### Gray scale

![Convert to gray scale](images/md/gray_scale.png)

#### Image 390 x 520 pixel

Grid dimension: 500 x 1 x 1

Block dimension: 256 x 1 x 1

Performance:

* **CPU**: 0.001107 second
* **GPU**: 0.00006508 second (17 times faster)

Metrics:

* **branch efficiency:** 100%
* **achieved occupancy:** 0.907
* **global memory load efficiency:** 100%
* **global memory store efficiency:** 100%

#### Image 3456 x 4608

Grid dimension: 1024 x 1 x 1

Block dimension: 1024 x 1 x 1

Performance:

* **CPU:** 0.1159 second
* **GPU:** 0.003149 second (37 time faster)

Metrics:

- **branch efficiency:** 100%
- **achieved occupancy:** 0.924
- **global memory load efficiency:** 100%
- **global memory store efficiency:** 100%

### Gaussian Filter

![Gaussian filter](images/md/gaussian_filter.png)

#### Image 390 x 520 pixel

Grid dimension: 300 x 1 x 1

Block dimension: 256 x 1 x 1

Performance:

- **CPU**: 0.02162 second
- **GPU**: 0.0002429 second (89 times faster)

Metrics:

- **branch efficiency:** 100%
- **achieved occupancy:** 0.931
- **global memory load efficiency:** 51.1%
- **global memory store efficiency:** 12.5%

#### Image 3456 x 4608

Grid dimension: 1024 x 1 x 1

Block dimension: 1024 x 1 x 1

Performance:

- **CPU:** 1.1436 second
- **GPU:** 0.05876 second (19 time faster)

Metrics:

- **branch efficiency:** 100%
- **achieved occupancy:** 0.915
- **global memory load efficiency:** 51.4%
- **global memory store efficiency:** 12.5%



