#!/usr/bin/env python3
"""make_synthetic.py -- a tiny deterministic ONNX model for hello_inference.

input  "input"  float32 [1, 32, 32, 3]   (NHWC, like the MediaPipe models)
  -> Transpose(perm=[0,3,1,2])            (NCHW for Conv)
  -> Conv(4 out channels, 3x3, pad 1)     (weights: (k % 7 - 3) / 16, bias: k / 8 - 0.25)
  -> Relu
output "output" float32 [1, 4, 32, 32]

Only the `onnx` package is needed (no TensorFlow). The weights are
integers over a power of two, so the CPU EP and the WebGPU EP differ
only by accumulation order.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_CH, IN_CH, K = 4, 3, 3
H = W = 32

w = np.array(
    [(k % 7 - 3) / 16.0 for k in range(OUT_CH * IN_CH * K * K)], dtype=np.float32
)
w = w.reshape(OUT_CH, IN_CH, K, K)
b = np.array([k / 8.0 - 0.25 for k in range(OUT_CH)], dtype=np.float32)

graph = helper.make_graph(
    [
        helper.make_node("Transpose", ["input"], ["nchw"], perm=[0, 3, 1, 2]),
        helper.make_node(
            "Conv", ["nchw", "W", "B"], ["conv"], kernel_shape=[K, K], pads=[1, 1, 1, 1]
        ),
        helper.make_node("Relu", ["conv"], ["output"]),
    ],
    "synthetic_conv",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, H, W, IN_CH])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, OUT_CH, H, W])],
    initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "synthetic_conv.onnx")
print("wrote synthetic_conv.onnx")
