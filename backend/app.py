import modal
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.2.0", "fastapi", "uvicorn", "numpy")
    .add_local_file("kernels/fused_relu_bias.cu", "/root/kernels/fused_relu_bias.cu")
)

app = modal.App("cuda-kernel-lab", image=image)
web = FastAPI()
web.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.function(gpu="T4", timeout=180)
@modal.asgi_app()
def serve():
    return web

@web.on_event("startup")
async def compile_kernel():
    import subprocess, os
    os.makedirs("/root/build", exist_ok=True)
    result = subprocess.run([
        "nvcc", "-O3", "-shared", "-Xcompiler", "-fPIC",
        "-o", "/root/build/fused_relu_bias.so",
        "/root/kernels/fused_relu_bias.cu"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("NVCC ERROR:", result.stderr)
    else:
        print("Kernel compiled OK")

@web.post("/benchmark")
async def benchmark(request: Request):
    import ctypes, torch, numpy as np

    item       = await request.json()
    rows       = min(int(item.get("rows", 1024)), 8192)
    cols       = min(int(item.get("cols", 1024)), 8192)
    block_size = int(item.get("block_size", 256))

    lib = ctypes.CDLL("/root/build/fused_relu_bias.so")
    lib.launch_fused.restype   = ctypes.c_float
    lib.launch_unfused.restype = ctypes.c_float

    device = torch.device("cuda")
    x    = torch.randn(rows, cols, device=device)
    bias = torch.randn(cols,       device=device)
    out1 = torch.empty_like(x)
    out2 = torch.empty_like(x)
    tmp  = torch.empty_like(x)

    ptr = lambda t: ctypes.cast(t.data_ptr(), ctypes.c_void_p)

    for _ in range(3):
        lib.launch_fused(ptr(x), ptr(bias), ptr(out1),
                         ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_int(block_size))

    fused_times = []
    for _ in range(10):
        ms = lib.launch_fused(ptr(x), ptr(bias), ptr(out1),
                              ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_int(block_size))
        fused_times.append(ms)

    unfused_times = []
    for _ in range(10):
        ms = lib.launch_unfused(ptr(x), ptr(bias), ptr(tmp), ptr(out2),
                                ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_int(block_size))
        unfused_times.append(ms)

    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    _ = torch.relu(x + bias)
    e.record()
    torch.cuda.synchronize()
    pytorch_ms = s.elapsed_time(e)

    fused_ms   = float(np.mean(fused_times))
    unfused_ms = float(np.mean(unfused_times))

    bytes_fused   = (2 * rows * cols + cols) * 4
    bytes_unfused = (3 * rows * cols + cols) * 4
    bw_fused      = bytes_fused   / fused_ms   * 1e-6
    bw_unfused    = bytes_unfused / unfused_ms * 1e-6

    return {
        "rows": rows, "cols": cols, "block_size": block_size,
        "fused_ms":        round(fused_ms, 4),
        "unfused_ms":      round(unfused_ms, 4),
        "pytorch_ms":      round(pytorch_ms, 4),
        "speedup":         round(unfused_ms / fused_ms, 3),
        "bw_fused_gb_s":   round(bw_fused, 2),
        "bw_unfused_gb_s": round(bw_unfused, 2),
        "peak_bw_gb_s":    300,
        "fused_bw_util":   round(bw_fused / 300 * 100, 2),
    }