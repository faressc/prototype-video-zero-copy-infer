#!/usr/bin/env bash
# convert.sh -- MediaPipe .tflite -> .onnx via tf2onnx, in a throwaway
# Python 3.12 venv (TensorFlow has no 3.13+/3.14 wheels). Needs `uv`.
#
#   ./models/convert.sh            # downloads the .tflite files, converts
#
# The resulting .onnx files are committed; this script only has to run
# again when the upstream models change.
set -euo pipefail
cd "$(dirname "$0")"

MODELS="palm_detection_lite palm_detection_full hand_landmark_lite hand_landmark_full"
BASE_URL="https://storage.googleapis.com/mediapipe-assets"

if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python "tensorflow" "tf2onnx" "onnx"
fi

for m in $MODELS; do
    [ -f "$m.tflite" ] || curl -sSfL -o "$m.tflite" "$BASE_URL/$m.tflite"
    .venv/bin/python -m tf2onnx.convert --tflite "$m.tflite" --opset 17 --output "$m.onnx"
done

.venv/bin/python - <<'EOF'
import onnx
for m in ["palm_detection_lite", "palm_detection_full", "hand_landmark_lite", "hand_landmark_full"]:
    g = onnx.load(f"{m}.onnx").graph
    def shape(v): return [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
    print(m, "inputs", [(i.name, shape(i)) for i in g.input],
          "outputs", [(o.name, shape(o)) for o in g.output])
EOF
