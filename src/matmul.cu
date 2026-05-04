#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
            << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

__global__ void matmul(const float* d_A, const float* d_B, float* d_C, int N, int M, int K) 
{
    //
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y  * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        float sum = 0;

        for (int k = 0; k < K; ++k)
        {
            //multiply and accumulate
            sum += d_A[row * K + k] * d_B[k * N + col];
        }
        d_C[row * N + col] = sum;
    }
}

int main() 
{
    size_t rows = 1000, cols = 1000;
    //Total memory represented by size of matrix x quantity x 4 bytes
    size_t size = (rows * cols) * sizeof(float); 

    std::vector<float> h_A(rows * cols);
    std::vector<float> h_B(rows * cols);
    std::vector<float> h_C(rows * cols);


    try 
    {
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                h_A[i * cols + j] = static_cast<int>(i * cols + j + 1);
                h_B[i * cols + j] = static_cast<int>(i * cols + j + 2);
            }
        }
    } catch (const std::out_of_range& e) {
        std::cerr << "Index out of range: " << e.what() << '\n';
    }

    std::cout << sizeof(float) * rows * cols << "bytes for matrix A";
    std::cout << sizeof(float) * rows * cols << "bytes for matrix B";

    //Allocate memory on the GPU

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size)); //Square matricies multiplied together have same sized output matrix

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
            (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
            );

    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, cols, rows, cols);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            std::cout << h_C[r * cols + c] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
