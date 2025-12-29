import tensorrt as trt
import numpy as np
import cuda
import time
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ONNX_PATH = "model.onnx"
ENGINE_PATH = "model_fp16.plan"


# ---------------------------------------------------------
# Build FP16 TensorRT engine
# ---------------------------------------------------------
def build_engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print("→ Loading ONNX")
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("Parser:", parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    config.max_workspace_size = 4 * 1024**3  # 4GB

    print("→ Building engine…")
    engine = builder.build_engine(network, config)
    with open(ENGINE_PATH, "wb") as f:
        f.write(engine.serialize())

    print("✔ Engine saved:", ENGINE_PATH)


# ---------------------------------------------------------
# Load engine
# ---------------------------------------------------------
def load_engine():
    trt_runtime = trt.Runtime(TRT_LOGGER)
    with open(ENGINE_PATH, "rb") as f:
        return trt_runtime.deserialize_cuda_engine(f.read())


# ---------------------------------------------------------
# Allocate buffers
# ---------------------------------------------------------
def allocate_buffers(engine, batch_size=3):
    context = engine.create_execution_context()

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        idx = engine.get_binding_index(binding)
        shape = list(engine.get_binding_shape(binding))

        if shape[0] == -1:       # dynamic batch
            shape[0] = batch_size
            context.set_binding_shape(idx, shape)

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        size = np.prod(shape)

        host = cuda.pagelocked_empty(size, dtype)
        device = cuda.mem_alloc(host.nbytes)

        bindings.append(int(device))

        if engine.binding_is_input(binding):
            inputs.append((host, device, shape))
        else:
            outputs.append((host, device, shape))

    return context, bindings, inputs, outputs, stream


# ---------------------------------------------------------
# One inference
# ---------------------------------------------------------
def infer(context, bindings, inputs, outputs, stream):
    for host, device, _ in inputs:
        cuda.memcpy_htod_async(device, host, stream)

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    for host, device, _ in outputs:
        cuda.memcpy_dtoh_async(host, device, stream)

    stream.synchronize()


# ---------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------
def run_test():
    if not os.path.exists(ENGINE_PATH):
        build_engine()

    engine = load_engine()
    context, bindings, inputs, outputs, stream = allocate_buffers(engine, batch_size=3)

    # Fill inputs with random data
    for host, _, shape in inputs:
        host[:] = np.random.rand(*shape).astype(np.float16).flatten()

    # warmup
    for _ in range(10):
        infer(context, bindings, inputs, outputs, stream)

    # benchmark
    iters = 50
    t0 = time.time()
    for _ in range(iters):
        infer(context, bindings, inputs, outputs, stream)

    dt = (time.time() - t0) / iters
    print(f"\n🔥 FP16 inference time: {dt*1000:.2f} ms")
    print(f"🚀 FPS: {1/dt:.2f}")

    _, _, out_shape = outputs[0]
    print("Output shape:", out_shape)


if __name__ == "__main__":
    run_test()
