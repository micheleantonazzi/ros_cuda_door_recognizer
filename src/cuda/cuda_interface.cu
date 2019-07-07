#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_interface.h"

__global__ void test_kernel(){
    printf("Hello from GPU\n");
}

void CudaInterface::test_cuda(){
    test_kernel<<<1, 10>>>();
    cudaDeviceSynchronize();
}