#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_interface.h"

__global__ void test_kernel(){
    printf("ciao\n");
    unsigned int threadId = threadIdx.x;
    unsigned int totThread = gridDim.x * blockDim.x;


    printf("%i\n", totThread);

    __syncthreads();
    unsigned int operationNumber = 10 / totThread;
    for(int i = 0; i < operationNumber; ++i)
        printf("operation\n");

}

void CudaInterface::test_cuda(){
    test_kernel<<<1, 10>>>();
    cudaDeviceSynchronize();
}