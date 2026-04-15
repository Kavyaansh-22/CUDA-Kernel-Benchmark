/*
 * fused_relu_bias.cu
 * ------------------
 * Custom CUDA kernel: fused ReLU + bias add
 *
 * Why fused?
 * ----------
 * Naive approach: two separate kernels
 *   1. out = x + bias   (memory read + write)
 *   2. out = relu(out)  (memory read + write)
 * = 4 memory transactions
 *
 * Fused approach: one kernel
 *   1. out = relu(x + bias)  (memory read + write)
 * = 2 memory transactions
 * → ~2x memory bandwidth saving for memory-bound ops
 *
 * This is exactly the kind of kernel fusion NVIDIA's
 * compiler (nvFuser) and TensorRT do automatically.
 * Here we do it manually to understand the mechanism.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ─────────────────────────────────────────
// KERNEL: fused ReLU + bias
// Each thread handles one element
// bias is 1D (size = cols), broadcast across rows
// ─────────────────────────────────────────
__global__ void fused_relu_bias_kernel(
    const float* __restrict__ input,   // [rows x cols]
    const float* __restrict__ bias,    // [cols]
    float*       __restrict__ output,  // [rows x cols]
    int rows,
    int cols)
{
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        int col = idx % cols;                    // which bias to use
        float val = input[idx] + bias[col];      // bias add
        output[idx] = val > 0.0f ? val : 0.0f;  // ReLU (branchless)
    }
}

// ─────────────────────────────────────────
// KERNEL: naive bias add (unfused, for comparison)
// ─────────────────────────────────────────
__global__ void bias_add_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        output[idx] = input[idx] + bias[idx % cols];
    }
}

__global__ void relu_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

// ─────────────────────────────────────────
// HOST LAUNCHER: fused
// ─────────────────────────────────────────
extern "C" float launch_fused(
    const float* d_input,
    const float* d_bias,
    float*       d_output,
    int rows, int cols,
    int block_size)
{
    int total = rows * cols;
    int grid  = (total + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fused_relu_bias_kernel<<<grid, block_size>>>(d_input, d_bias, d_output, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// ─────────────────────────────────────────
// HOST LAUNCHER: unfused (2 kernels)
// ─────────────────────────────────────────
extern "C" float launch_unfused(
    const float* d_input,
    const float* d_bias,
    float*       d_intermediate,
    float*       d_output,
    int rows, int cols,
    int block_size)
{
    int total = rows * cols;
    int grid  = (total + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bias_add_kernel<<<grid, block_size>>>(d_input, d_bias, d_intermediate, rows, cols);
    relu_kernel<<<grid, block_size>>>(d_intermediate, d_output, total);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}
