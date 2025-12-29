import onnxruntime as ort
import numpy as np
import time

# ---------------------------------------------------------
# 加载 ONNX 模型（FP16）
# ---------------------------------------------------------
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 优化 EP（TensorRT 没装就不管，优先 CUDA、然后 CPU）
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 0,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
]

print("Loading ONNX...")
session = ort.InferenceSession("model.onnx", sess_options, providers=providers)

# ---------------------------------------------------------
# 构造 FP16 输入数据
# ---------------------------------------------------------
rsm_shape = (3, 7, 384, 64)
surf_shape = (3,10,512,512)

input_rsm = (np.random.rand(*rsm_shape).astype(np.float16))
input_surf = (np.random.rand(*surf_shape).astype(np.float16))

# ---------------------------------------------------------
# warmup
# ---------------------------------------------------------
print("Warmup...")
for _ in range(5):
    session.run(
        None,
        {
            "input_rsm": input_rsm,
            "input_surface": input_surf
        }
    )

# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------
iters = 30
print("Benchmark start...")

t0 = time.time()
for _ in range(iters):
    session.run(
        None,
        {
            "input_rsm": input_rsm,
            "input_surface": input_surf
        }
    )
dt = (time.time() - t0) / iters

print(f"\n🔥 ONNX Runtime FP16 Inference: {dt*1000:.2f} ms")
print(f"🚀 FPS: {1.0/dt:.2f}")

# ---------------------------------------------------------
# 输出 shape 检查
# ---------------------------------------------------------
out = session.run(
    None,
    {
        "input_rsm": input_rsm,
        "input_surface": input_surf
    }
)[0]

print("Output shape:", out.shape)
