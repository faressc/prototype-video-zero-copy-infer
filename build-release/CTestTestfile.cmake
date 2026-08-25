# CMake generated Testfile for 
# Source directory: /home/faressc/concepts/simple-wayland-window
# Build directory: /home/faressc/concepts/simple-wayland-window/build-release
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[infer_cpu_cpu]=] "/home/faressc/concepts/simple-wayland-window/build-release/hello_inference" "--src" "cpu" "--ep" "cpu" "--iters" "3" "--strict" "--model" "/home/faressc/concepts/simple-wayland-window/models/synthetic_conv.onnx")
set_tests_properties([=[infer_cpu_cpu]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/faressc/concepts/simple-wayland-window/CMakeLists.txt;310;add_test;/home/faressc/concepts/simple-wayland-window/CMakeLists.txt;0;")
add_test([=[infer_all_synthetic]=] "/home/faressc/concepts/simple-wayland-window/build-release/hello_inference" "--all" "--iters" "5" "--strict" "--model" "/home/faressc/concepts/simple-wayland-window/models/synthetic_conv.onnx")
set_tests_properties([=[infer_all_synthetic]=] PROPERTIES  LABELS "gpu" _BACKTRACE_TRIPLES "/home/faressc/concepts/simple-wayland-window/CMakeLists.txt;313;add_test;/home/faressc/concepts/simple-wayland-window/CMakeLists.txt;0;")
