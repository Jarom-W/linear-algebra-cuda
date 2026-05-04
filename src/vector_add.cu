#include <iostream>
#include <vector>
#include <cuda_runtime.h>

//Vector addition kernel

__global__ void vector_add(const float* A, const float* B, float* C, int N) {

    //the index of each thread is calculdated this way
    //i = (which block i'm in * threads per block) + my position in block
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<N) {
        C[i]  = A[i] + B[i];
    }
}

int main() {
    int N = 1000; //Number of indicies
    size_t size = N * sizeof(float);

    std::vector<float> h_A(N), h_B(N), h_C(N); //h = host. Host vectors living in CPU RAM

    //Initialize data
    for (int i=0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }
    
    //Allocate space for vectors
    //d = device. Lives in GPU VRAM
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    //Copy memory to GPU
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    //Launch kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    //Wait for GPU
    cudaDeviceSynchronize();

    //Copy back from GPU to CPU
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    //Verify results

    for (int i = 0; i < 5; i++)
    {
        std::cout << h_A[i] << " + " << h_B[i]
            << " = " << h_C[i] << std::endl;
    }
    
    //Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

};
