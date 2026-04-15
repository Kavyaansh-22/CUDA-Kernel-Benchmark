CUDA Performance Frontier: Custom Kernels vs. Frameworks
This project is a high-performance benchmarking suite designed to profile the efficiency of custom CUDA C kernels against industry-standard PyTorch implementations. It provides a real-time comparison of execution speed, memory bandwidth utilization, and arithmetic intensity across local and cloud-based NVIDIA hardware.

The Core Mission:
While high-level frameworks are convenient, they often carry overhead. This project demonstrates:

Custom Kernel Engineering: Implementation of a high-performance fused ReLU + Bias kernel written in raw CUDA C.

The "Fusion" Advantage: Proving how merging multiple operations into a single GPU kernel pass reduces memory bottlenecks.

Hardware Profiling: Real-time analysis on NVIDIA T4 (Cloud) via Modal and future integration for Local/Edge hardware.



Tech Stack:
GPU Backend: CUDA C++, NVIDIA T4 (Cloud), PyTorch.
Infrastructure: Modal for serverless GPU execution and FastAPI for the benchmarking endpoint.
Visualization: Live "Hacker-style" dashboard built with Vanilla JS and Chart.js.
Profiling: High-precision cudaEvent_t timing and Roofline Model analysis.



📊 Key Metrics Tracked:
Throughput (TFLOPS): Measures raw computational speed.
Bandwidth Utilization (GB/s): Tracks how effectively the kernel uses the GPU's memory bus.
Arithmetic Intensity: Categorizes kernels as either Compute-Bound or Memory-Bound.
Speedup Ratio: Direct comparison (e.g., 1.9x) of custom kernels vs. standard PyTorch.

Project Structure:
.
├── backend/                # Serverless GPU Infrastructure
│   ├── app.py              # FastAPI & Modal benchmarking logic
│   └── requirements.txt    # Python dependencies
├── frontend/               # Performance Dashboard
│   ├── index.html          # UI with Chart.js visualization
│   └── assets/             # Styles and frontend logic
├── kernels/                # Raw CUDA Engineering
│   ├── fused_relu_bias.cu  # Custom C++ CUDA implementation
│   └── fused_relu_bias.c   # CPU-based reference implementation
└── readme.md               # Documentation and Roadmap


Future Roadmap:
Local vs. Cloud Latency: Adding deep-link measurements for data transfer overhead.
Edge Integration: Running benchmarks on Raspberry Pi and NVIDIA Jetson to visualize "The Edge Gap."
Fused Library: Expanding to more complex fused operations like LayerNorm and Softmax.

Author: Kavyaansh Kundu
Status: Initial full project commit pushed to main.
