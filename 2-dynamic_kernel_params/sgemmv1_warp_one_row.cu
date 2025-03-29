#include <cstdio>
#include "sgemm_utils.h"
#include <chrono>
#include <iostream>
void random_matrix(int m, int n, int k, float *A, float *B)
{
    int lda = k;
    int ldb = n;
    // A matrix init
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            A(i, j) = 2.0 * drand48() - 1.0;
            // A(i, j) = 1.0;
        }
    }
    // B matrix init
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            B(i, j) = 2.0 * drand48() - 1.0;
            // B(i, j) = 1.0;
        }
    }
}

void compare_matrices(int M, int N, float *A, float *B)
{
    float max_diff = 0.f, diff;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            diff = abs(A[i * N + j] - B[i * N + j]);
            max_diff = (diff > max_diff ? diff : max_diff);
        }
    }
    printf("max diff = %f \n", max_diff);
}

void cpu_sgemm(int M, int N, int K, float *A, float *B, float *C)
{
    int lda = K;
    int ldb = N;
    int ldc = N;

    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float temp = 0.f;
            for (int k = 0; k < K; k++)
            {
                temp += A(m, k) * B(k, n);
            }
            C(m, n) = temp;
        }
    }

    printf("%f\n", C(0, 0));
}

__device__ inline float warp_reduce_sum(float val)
{
    for (int offset = 32 / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    return val;
}

template <int Blocksize>
__global__ void sgemm(int M, int N, int K, float *A, float *B, float *C)
{

    int offset = threadIdx.x;
    const int x = blockIdx.x;
    const int y = blockIdx.y;

    float *A_ptr_start = A + blockIdx.y * K;
    float *B_ptr_start = B + blockIdx.x;
    float temp = 0.f;
    for (int k = offset; k < K; k += Blocksize)
    {
        temp += A_ptr_start[k] * B_ptr_start[k * N]; // 注意 k 是A的列，B的行
    }

    C[x + y * N] = warp_reduce_sum(temp);
}

int main()
{
    int M = 512;
    int N = 512;
    int K = 512;

    const size_t mem_size_A = M * K * sizeof(float);
    const size_t mem_size_B = K * N * sizeof(float);
    const size_t mem_size_C = M * N * sizeof(float);

    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);

    float *matrix_C_gpu_host = (float *)malloc(mem_size_C);
    float *matrix_C_cpu_host = (float *)malloc(mem_size_C);

    random_matrix(M, N, K, matrix_A_host, matrix_B_host);
    cpu_sgemm(M, N, K, matrix_A_host, matrix_B_host, matrix_C_cpu_host);

    float *matrix_A_device, *matrix_B_device, *matrix_C_device;
    // 获取释放前的内存信息
    size_t freeMemBefore, totalMem;
    cudaMemGetInfo(&freeMemBefore, &totalMem);
    printf("Before freeing: Free %lu, Total: %lu\n", freeMemBefore, totalMem);

    cudaMalloc((void **)&matrix_A_device, mem_size_A);
    cudaMalloc((void **)&matrix_B_device, mem_size_B);
    cudaMalloc((void **)&matrix_C_device, mem_size_C);

    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);

    const int blocksize = 32;
    dim3 Block(blocksize, blocksize);
    dim3 Grid(M, N);

    sgemm<blocksize><<<Grid, Block>>>(M, N, K, matrix_A_device, matrix_B_device, matrix_C_device);

    cudaMemcpy(matrix_C_gpu_host, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    printf("%f\n", matrix_C_gpu_host[0]);
    compare_matrices(M, N, matrix_C_cpu_host, matrix_C_gpu_host);

    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_cpu_host);
    free(matrix_C_gpu_host);
    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
    std::cout << "free device buffer" << std::endl;
    // 获取释放后的内存信息
    size_t freeMemAfter;
    cudaMemGetInfo(&freeMemAfter, &totalMem);
    printf("After freeing: Free %lu, Total: %lu\n", freeMemAfter, totalMem);

    return 0;
}