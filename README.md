# GPU Kernel Profiler & CUDA Performance Lab 🚀

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen?style=for-the-badge&logo=vercel)](https://cuda-kernel-benchmark.vercel.app/)
[![NVIDIA T4](https://img.shields.io/badge/Hardware-NVIDIA%20T4-76B900?style=for-the-badge&logo=nvidia)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

A high-performance computing (HPC) laboratory focused on benchmarking custom C++ fused kernels against industry-standard deep learning frameworks. This project explores the performance delta between local execution and cloud-based NVIDIA T4 GPUs.

## 🔬 Project Overview

This lab implements a **low-level Fused ReLU + Bias Add kernel** from scratch in CUDA C. By fusing these operations, we minimize global memory round-trips, significantly reducing the bottleneck caused by memory bandwidth limits in deep learning workloads.

### Key Features
* **Custom Kernel Engineering**: Hand-written `__global__` CUDA kernels implementing operator fusion.
* **Real-time Roofline Analysis**: Automatic calculation of Arithmetic Intensity (FLOP/byte) to determine if workloads are Compute-bound or Memory-bound.
* **Hardware Profiling**: High-precision performance tracking using `cudaEvent_t` for timing and `torch.cuda.memory_stats` for peak allocation tracking.
* **Hybrid Infrastructure**: Serverless GPU execution via Modal (T4) paired with a real-time profiling dashboard.

## 📂 Project Structure

```bash
.
├── backend/                # Serverless GPU Infrastructure
│   ├── app.py              # FastAPI & Modal benchmarking logic
│   └── requirements.txt    # Python dependencies
├── frontend/               # Performance Dashboard
│   ├── index.html          # UI with Chart.js visualization
│   └── assets/             # Styles and frontend logic
├── kernels/                # Raw CUDA Engineering
│   ├── fused_relu_bias.cu  # Parallel CUDA implementation (High Performance)
│   └── fused_relu_bias.c   # Sequential CPU reference (Verification Baseline)
└── readme.md               # Documentation and Roadmap

📊 Performance Metrics
The dashboard visualizes critical hardware counters to provide an end-to-end performance profile:

Throughput (TFLOPS): Raw computational throughput compared to T4 peak theoretical performance.

Bandwidth Utilization: Efficiency of data movement across the memory bus (GB/s).

The "Fusion" Speedup: Direct comparison between standard unfused PyTorch operations and the custom fused C++ implementation.

🛠 Tech Stack
Languages: CUDA C++, Python, JavaScript.

Frameworks: PyTorch (Baseline), FastAPI.

Hardware: NVIDIA T4 (Cloud), Local CPU (Baseline).

Deployment: Modal (Backend), Vercel (Frontend).

🏁 Future Roadmap
[ ] Edge Integration: Porting benchmarks to NVIDIA Jetson to analyze Unified Memory performance.

[ ] Vectorized Access: Implementing float4 memory access for enhanced throughput.

[ ] Advanced Fusion: Developing fused LayerNorm and Softmax kernels for LLM-specific workloads.

Author: Kavyaansh Kundu
