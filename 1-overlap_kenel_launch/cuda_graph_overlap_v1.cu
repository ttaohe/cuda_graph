#include <cuda_runtime.h>
#include <iostream>
#include <chrono> // 用于计时

// 定义宏
#define NSTEP 1000
#define NKERNEL 20

// 定义一个简单的CUDA内核函数
__global__ void shortKernel(float *out_d, const float *in_d, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算全局线程索引
    if (idx < size)
    {
        out_d[idx] = in_d[idx] * 2.0f; // 示例操作：将输入数据乘以2
    }
}

int main()
{
    const int size = 500000; // 数据大小
    const int bytes = size * sizeof(float);

    // 分配GPU内存
    float *in_d;
    float *out_d;
    cudaMalloc(&in_d, bytes);
    cudaMalloc(&out_d, bytes);

    // 初始化输入数据
    float *in_h = new float[size];
    for (int i = 0; i < size; i++)
    {
        in_h[i] = static_cast<float>(i); // 示例初始化
    }
    cudaMemcpy(in_d, in_h, bytes, cudaMemcpyHostToDevice);

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动计时器
    auto start_time = std::chrono::high_resolution_clock::now();

    // 执行内核
    for (int istep = 0; istep < NSTEP; istep++)
    {
        for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++)
        {
            // 启动内核
            shortKernel<<<(size + 255) / 256, 256, 0, stream>>>(out_d, in_d, size);
        }
        // 同步流
        cudaStreamSynchronize(stream);
    }

    // 停止计时器
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Total execution time: " << duration << " ms" << std::endl;

    // 验证结果（可选）
    float *out_h = new float[size];
    cudaMemcpy(out_h, out_d, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++)
    { // 检查前10个结果
        std::cout << "out_h[" << i << "] = " << out_h[i] << std::endl;
    }

    // 释放资源
    cudaFree(in_d);
    cudaFree(out_d);
    delete[] in_h;
    delete[] out_h;
    cudaStreamDestroy(stream);

    return 0;
}