# export_onnx.py
import torch
from model_def import build_model

model = build_model().cuda().half()

rsm_dummy   = torch.randn(3, 7, 384, 64).cuda().half()
surf_dummy  = torch.randn(3,10,512,512).cuda().half()

torch.onnx.export(
    model,
    (rsm_dummy, surf_dummy),
    "model.onnx",
    input_names=["input_rsm", "input_surface"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes=None
)

print("✔ ONNX exported to model.onnx")
