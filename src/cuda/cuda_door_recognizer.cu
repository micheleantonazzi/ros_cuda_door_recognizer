#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void test_kernel(){
    printf("Hello form gpu");
}

void test_cuda(){
    test_kernel<<<1, 10>>>();
    cudaDeviceReset();
}