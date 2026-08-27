# models/

The ONNX models `hello_inference` and the hand scenes run, plus the
scripts that produced them. The `.onnx` files are committed so the build
never depends on a Python toolchain.

| file | input | outputs | origin |
|---|---|---|---|
| `synthetic_conv.onnx` | `input` float32 `[1,32,32,3]` (NHWC) | `output` `[1,4,32,32]` | `make_synthetic.py` (`onnx` package only): Transpose → Conv 3×3, 4 ch, deterministic weights `(k % 7 − 3)/16` → Relu. Small enough that every (source, EP) cell can be compared to 1e-4. |
| `palm_detection_lite.onnx`, `palm_detection_full.onnx` | `input_1` `[1,192,192,3]` | `Identity` `[1,2016,18]` (anchor regressors), `Identity_1` `[1,2016,1]` (scores) | MediaPipe `palm_detection_{lite,full}.tflite` via `convert.sh` (tf2onnx, opset 17) |
| `hand_landmark_lite.onnx`, `hand_landmark_full.onnx` | `input_1` `[1,224,224,3]` | `Identity` `[1,63]` (21 landmarks × xyz, pixels), `Identity_1` `[1,1]` (hand presence), `Identity_2` `[1,1]` (handedness), `Identity_3` `[1,63]` (world landmarks) | same |

The `hand_*` binaries and `hello_hand` default to the **`_full`** pair —
the weights MediaPipe's own web demo runs (`modelComplexity: 1` in the
JS solution; the Tasks `hand_landmarker.task` bundles the full landmark
model too). `--lite` selects the cheaper pair; the ctest entries pin it
explicitly so the committed golden values do not move. The lite landmark
model is visibly noisier around the 0.5 presence gate.

Regenerate with `./models/convert.sh` (needs `uv`; makes a throwaway
Python 3.12 venv in `models/.venv` because TensorFlow ships no wheels
for newer interpreters, downloads the `.tflite` files from
`storage.googleapis.com/mediapipe-assets`, converts, prints the I/O
shapes). `python3 models/make_synthetic.py` needs only `onnx`.

The MediaPipe models are Apache-2.0 (Google). The tf2onnx conversion
keeps the NHWC input and inserts transposes in front of the convolutions;
ONNX Runtime's WebGPU EP is told `preferredLayout=NHWC` so it can fold
them back. The CUDA EP's kernels are NCHW by default, so there the
transposes actually run; ORT carries NHWC CUDA kernels behind the
`prefer_nhwc=1` provider option (`INFER_ORT_OPTS=prefer_nhwc=1`) — worth
measuring before making it the default.
