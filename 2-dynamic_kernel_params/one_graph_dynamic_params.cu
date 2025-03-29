#include <cuda_runtime.h>
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

__device__ inline float warp_reduce_sum(float val)
{
    for (int offset = 32 / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    return val;
}

__global__ void sgemm(int M, int N, int K, float *A, float *B, float *C)
{

    int offset = threadIdx.x;
    const int x = blockIdx.x;
    const int y = blockIdx.y;

    float *A_ptr_start = A + blockIdx.y * K;
    float *B_ptr_start = B + blockIdx.x;
    float temp = 0.f;
    for (int k = offset; k < K; k += blockDim.x)
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

    cudaGraph_t cgraph;
    cudaStream_t cstream;
    cudaStreamCreate(&cstream);
    cudaGraphExec_t instance;

    // 开始在这个流里捕获图
    cudaStreamBeginCapture(cstream, cudaStreamCaptureModeGlobal);

    // func
    {
        cudaKernelNodeParams params;
        params.blockDim = {32, 1, 1};
        params.gridDim = {static_cast<unsigned>(M), static_cast<unsigned int>(N), 1};
        params.sharedMemBytes = 0;
        params.extra = nullptr;
        params.func = reinterpret_cast<void *>(sgemm);

        void *kenelParams[] = {
            &M,
            &N,
            &K,
            &matrix_A_device,
            &matrix_B_device,
            &matrix_C_device};

        params.kernelParams = kenelParams;
        cudaStreamCaptureStatus capture_status;
        const cudaGraphNode_t *deps;
        size_t dep_count;
        cudaStreamGetCaptureInfo_v2(cstream, &capture_status, nullptr, &cgraph, &deps, &dep_count);

        cudaGraphNode_t new_node;
        cudaGraphAddKernelNode(&new_node, cgraph, deps, dep_count, &params);
    }

    cudaStreamEndCapture(cstream, &cgraph);
    cudaGraphInstantiate(&instance, cgraph, nullptr, nullptr, 0);

    cudaGraphLaunch(instance, cstream);

    cudaMemcpy(matrix_C_gpu_host, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    printf("%f\n", matrix_C_gpu_host[0]);

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