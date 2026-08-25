cmake_minimum_required(VERSION 3.20)
project(hello_wayland C)

# Layout:
#   common/    the Wayland client layer every binary shares (README §4-§13)
#   hello/     the four single-file references: one window, rendered four
#              ways -- shm, GLES/EGLSurface, GLES/dmabuf, Vulkan (§14-§16)
#   present/   stage two: presenter libraries extracted from hello/ --
#              context + swapchain + sync + present loop, scene-agnostic (§17)
#   scenes/    what gets drawn on top of a presenter (the cube)
#   shaders/   GLSL for the Vulkan binaries, compiled to SPIR-V at build time
#   camera/    stage three: V4L2 capture exporting dmabufs (§18)
#   infer/     stage four: tensors across domains + ONNX Runtime (§20)
#   models/    the ONNX models and the scripts that produced them

# ---------------------------------------------------------------------------
# 1. Find the libraries.
#    wayland-client  = libwayland (socket handling, proxies, dispatch)
#    wayland-protocols = a directory of protocol XML files (no code!)
#    wayland-scanner = the code generator that turns XML into C
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(WAYLAND_CLIENT REQUIRED IMPORTED_TARGET wayland-client)
# xkbcommon: translates raw keyboard scancodes into symbols using the
# keymap the compositor sends us. Keyboard input is unusable without it.
pkg_check_modules(XKBCOMMON REQUIRED IMPORTED_TARGET xkbcommon)
# wayland-cursor: loads the XCursor theme from disk and wraps the images
# in wl_buffers -- the cursor itself is just another surface + buffer.
pkg_check_modules(WAYLAND_CURSOR REQUIRED IMPORTED_TARGET wayland-cursor)

pkg_get_variable(WAYLAND_PROTOCOLS_DIR wayland-protocols pkgdatadir)
pkg_get_variable(WAYLAND_SCANNER wayland-scanner wayland_scanner)

# ---------------------------------------------------------------------------
# 2. Protocol bindings. add_wayland_protocol(<target> <xml-path-relative-to
#    wayland-protocols>): two scanner runs -- `client-header` (the inline
#    request wrappers) + `private-code` (the wl_interface tables, hidden
#    visibility) -- into a static library that carries its include dir.
#    README §1-§2 on why this is generated per app instead of shipped as
#    a shared library.
# ---------------------------------------------------------------------------
function(add_wayland_protocol TARGET XML_REL)
    set(XML ${WAYLAND_PROTOCOLS_DIR}/${XML_REL})
    get_filename_component(BASENAME ${XML_REL} NAME_WE)
    set(HDR ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}-client-protocol.h)
    set(SRC ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}-protocol.c)
    add_custom_command(
        OUTPUT ${HDR}
        COMMAND ${WAYLAND_SCANNER} client-header ${XML} ${HDR}
        DEPENDS ${XML})
    add_custom_command(
        OUTPUT ${SRC}
        COMMAND ${WAYLAND_SCANNER} private-code ${XML} ${SRC}
        DEPENDS ${XML})
    add_library(${TARGET} STATIC ${SRC} ${HDR})
    target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${TARGET} PUBLIC PkgConfig::WAYLAND_CLIENT)
endfunction()

# the window role (§5)
add_wayland_protocol(xdg_shell_protocol stable/xdg-shell/xdg-shell.xml)
# touchpad pinch/swipe/hold as events; the pinch drives the cube's zoom (§7)
add_wayland_protocol(pointer_gestures_protocol
                     unstable/pointer-gestures/pointer-gestures-unstable-v1.xml)
# GPU buffers as wl_buffers, bound by US in the dmabuf backends (§14)
add_wayland_protocol(dmabuf_protocol stable/linux-dmabuf/linux-dmabuf-v1.xml)
# explicit sync: acquire/release timeline points on the surface (§14)
add_wayland_protocol(syncobj_protocol staging/linux-drm-syncobj/linux-drm-syncobj-v1.xml)

# ---------------------------------------------------------------------------
# 3. app_common -- the shared Wayland client layer (registry, xdg role +
#    configure/ack, seat input with xkbcommon + pinch gestures, themed
#    cursor, animation clock, frame-callback loop). Backends provide two
#    hooks.
# ---------------------------------------------------------------------------
add_library(app_common STATIC common/common.c common/common.h)
target_include_directories(app_common PUBLIC common)
target_link_libraries(app_common PUBLIC xdg_shell_protocol
                                        pointer_gestures_protocol
                                        PkgConfig::XKBCOMMON
                                        PkgConfig::WAYLAND_CURSOR
                                        m) # fmod in the animation clock

# ---------------------------------------------------------------------------
# 4. The four references in hello/: the same window, four renderers.
# ---------------------------------------------------------------------------

# CPU rendering into wl_shm buffers.
add_executable(hello_wayland hello/shm.c)
target_link_libraries(hello_wayland PRIVATE app_common)

# GLES through the CLASSIC path: wl_egl_window + EGLSurface + eglSwapBuffers,
# swapchain hidden inside Mesa (how GTK/Qt/weston-simple-egl work).
pkg_check_modules(GLES_EGLSURFACE_STACK REQUIRED IMPORTED_TARGET egl glesv2 wayland-egl)
add_executable(hello_wayland_gles_eglsurface hello/gles-eglsurface.c)
target_link_libraries(hello_wayland_gles_eglsurface PRIVATE app_common
                                                            PkgConfig::GLES_EGLSURFACE_STACK)

# Raw GLES + a hand-rolled dmabuf swapchain: EGL kept ONLY as context
# factory (surfaceless); gbm_bo allocation + linux-dmabuf + syncobj, bound
# by US. GLSL still compiles AT RUNTIME inside the driver.
pkg_check_modules(GLES_DMABUF_STACK REQUIRED IMPORTED_TARGET egl glesv2 gbm libdrm)
add_executable(hello_wayland_gles_dmabuf hello/gles-dmabuf.c)
target_link_libraries(hello_wayland_gles_dmabuf PRIVATE app_common dmabuf_protocol
                                                        syncobj_protocol
                                                        PkgConfig::GLES_DMABUF_STACK)

# Vulkan. Shaders compile AT BUILD TIME to SPIR-V -- the same pattern as
# wayland-scanner: a spec (GLSL) is compiled into an artifact (.spv) the
# runtime then merely loads. Needs glslc or glslangValidator.
pkg_check_modules(VULKAN REQUIRED IMPORTED_TARGET vulkan)
find_program(GLSLC glslc)
find_program(GLSLANG glslangValidator)

if(GLSLC OR GLSLANG)
    set(SHADER_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

    # add_spirv_shaders(<out-var> shader.vert shader.frag ...): one
    # custom command per GLSL file; the .spv paths land in <out-var> so
    # they can be listed as executable sources (Ninja then rebuilds them
    # when the GLSL changes, exactly like a .c file).
    # shared GLSL pulled in via #include (glslc resolves it relative to
    # the including file); listed as a dependency of every shader
    set(SHADER_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cam_effects.glsl
                        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/infer_hash.glsl)
    function(add_spirv_shaders OUT_VAR)
        set(SPVS)
        foreach(SHADER ${ARGN})
            set(SPV ${SHADER_OUT_DIR}/${SHADER}.spv)
            if(GLSLC)
                add_custom_command(
                    OUTPUT ${SPV}
                    COMMAND ${GLSLC} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER} -o ${SPV}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER} ${SHADER_INCLUDES})
            else()
                add_custom_command(
                    OUTPUT ${SPV}
                    COMMAND ${GLSLANG} -V -I${CMAKE_CURRENT_SOURCE_DIR}/shaders
                            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER} -o ${SPV}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER} ${SHADER_INCLUDES})
            endif()
            list(APPEND SPVS ${SPV})
        endforeach()
        set(${OUT_VAR} ${SPVS} PARENT_SCOPE)
    endfunction()

    add_spirv_shaders(GRADIENT_SPV gradient.vert gradient.frag)
    add_executable(hello_wayland_vulkan hello/vulkan.c ${GRADIENT_SPV})
    target_compile_definitions(hello_wayland_vulkan PRIVATE SHADER_DIR="${SHADER_OUT_DIR}")
    target_link_libraries(hello_wayland_vulkan PRIVATE app_common PkgConfig::VULKAN)
else()
    message(WARNING "No GLSL->SPIR-V compiler found: skipping the Vulkan targets. "
                    "Install one with: sudo dnf install glslc   (or: glslang)")
endif()

# ---------------------------------------------------------------------------
# 5. Stage two (README §17): presenters + scenes. Each GPU backend is split
#    at the seam "who owns the swapchain" vs "who draws": a presenter
#    library (extracted from hello/) and a scene that plugs into it via a
#    four-function vtable.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 6. Stage three (README §18): the camera. A raw V4L2 capture device
#    exporting its buffers as dmabufs, plus the pacing/lifetime helper;
#    scenes import the buffers zero-copy (GLES, Vulkan) or map them (CPU).
# ---------------------------------------------------------------------------
add_library(camera STATIC camera/v4l2_camera.c camera/v4l2_camera.h
                          camera/cam_stream.c camera/cam_stream.h)
target_include_directories(camera PUBLIC camera)

# embed_glsl(<header> <glsl-file> <symbol>): the GLES scenes compile
# shaders from source strings at runtime, so the shared effect file is
# turned into a C string at configure time (re-run when it changes).
function(embed_glsl OUT_HDR INPUT SYMBOL)
    file(READ ${INPUT} HEX_CONTENT HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," HEX_CONTENT ${HEX_CONTENT})
    file(WRITE ${OUT_HDR} "/* generated from ${INPUT} */\nstatic const char ${SYMBOL}[] = {${HEX_CONTENT}0x00};\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${INPUT})
endfunction()
embed_glsl(${CMAKE_CURRENT_BINARY_DIR}/cam_effects_glsl.h
           ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cam_effects.glsl cam_effects_glsl)

# the CPU presenter + the cube through a software rasterizer: every
# pipeline stage the GPU scenes declare in a struct, as a loop
add_library(shm_present STATIC present/shm_present.c present/shm_present.h)
target_include_directories(shm_present PUBLIC present)
target_link_libraries(shm_present PUBLIC app_common)

add_executable(cube_shm scenes/scene_cube_shm.c scenes/cube_mesh.h scenes/mat4.h)
target_link_libraries(cube_shm PRIVATE shm_present m)

pkg_check_modules(GLES_PRESENT_STACK REQUIRED IMPORTED_TARGET egl glesv2 gbm libdrm wayland-egl)
add_library(gles_present STATIC present/gles_present.c present/gles_present.h)
target_include_directories(gles_present PUBLIC present)
target_link_libraries(gles_present PUBLIC app_common dmabuf_protocol syncobj_protocol
                                          PkgConfig::GLES_PRESENT_STACK)

# one scene source, two binaries: the presenter mode is a compile-time pick
add_executable(cube_gles_dmabuf scenes/scene_cube_gles.c scenes/cube_mesh.h scenes/mat4.h)
target_compile_definitions(cube_gles_dmabuf PRIVATE CUBE_GLES_MODE=GLES_MODE_DMABUF)
target_link_libraries(cube_gles_dmabuf PRIVATE gles_present)

add_executable(cube_gles_eglsurface scenes/scene_cube_gles.c scenes/cube_mesh.h scenes/mat4.h)
target_compile_definitions(cube_gles_eglsurface PRIVATE CUBE_GLES_MODE=GLES_MODE_EGLSURFACE)
target_link_libraries(cube_gles_eglsurface PRIVATE gles_present)

if(GLSLC OR GLSLANG)
    add_library(vk_present STATIC present/vk_present.c present/vk_present.h)
    target_include_directories(vk_present PUBLIC present)
    target_link_libraries(vk_present PUBLIC app_common PkgConfig::VULKAN)

    add_spirv_shaders(CUBE_SPV cube.vert cube.frag)
    add_executable(cube_vulkan scenes/scene_cube_vk.c scenes/cube_mesh.h scenes/mat4.h ${CUBE_SPV})
    target_compile_definitions(cube_vulkan PRIVATE SHADER_DIR="${SHADER_OUT_DIR}")
    target_link_libraries(cube_vulkan PRIVATE vk_present)
endif()

# ---------------------------------------------------------------------------
# 7. The camera scenes: the same effect chain on the CPU, on GLES (zero-copy
#    via EGLImage + samplerExternalOES), and on Vulkan (zero-copy via
#    external memory + YCbCr sampler, CPU staging as the fallback).
# ---------------------------------------------------------------------------
add_executable(cam_shm scenes/scene_cam_shm.c scenes/raster.h scenes/cube_mesh.h scenes/mat4.h)
target_link_libraries(cam_shm PRIVATE shm_present camera m)

add_executable(cam_gles_dmabuf scenes/scene_cam_gles.c scenes/cube_mesh.h scenes/mat4.h)
target_compile_definitions(cam_gles_dmabuf PRIVATE CAM_GLES_MODE=GLES_MODE_DMABUF)
target_include_directories(cam_gles_dmabuf PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(cam_gles_dmabuf PRIVATE gles_present camera)

add_executable(cam_gles_eglsurface scenes/scene_cam_gles.c scenes/cube_mesh.h scenes/mat4.h)
target_compile_definitions(cam_gles_eglsurface PRIVATE CAM_GLES_MODE=GLES_MODE_EGLSURFACE)
target_include_directories(cam_gles_eglsurface PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(cam_gles_eglsurface PRIVATE gles_present camera)

pkg_check_modules(LIBDRM REQUIRED IMPORTED_TARGET libdrm) # drm_fourcc.h for the modifier
if(GLSLC OR GLSLANG)
    add_spirv_shaders(CAM_SPV cam.vert cam.frag cube_cam.vert cube_cam.frag)
    add_executable(cam_vulkan scenes/scene_cam_vk.c scenes/cube_mesh.h scenes/mat4.h ${CAM_SPV})
    target_compile_definitions(cam_vulkan PRIVATE SHADER_DIR="${SHADER_OUT_DIR}")
    target_link_libraries(cam_vulkan PRIVATE vk_present camera PkgConfig::LIBDRM)
endif()

# ---------------------------------------------------------------------------
# 8. Stage four (README §20): inference. `infer` is a headless library --
#    typeless tensors in four memory domains (CPU, GLES, Vulkan, WebGPU),
#    a probed registry of conversion edges with explicit cost classes,
#    and ONNX Runtime behind an engine seam. hello_inference sweeps the
#    (source domain x execution provider) matrix. WE own Dawn (the
#    anira v3 arrangement, doc section 4): DAWN_ROOT is a standalone
#    monolithic-shared Dawn install (libwebgpu_dawn.so) on which the
#    device is created; onnxruntime is built with
#    onnxruntime_USE_EXTERNAL_DAWN=ON and onnxruntime_CUSTOM_DAWN_SRC_PATH=
#    <the same Dawn source tree>, so it links only dawn_proc thunks and
#    receives our proc table (option "dawnProcTable") at session creation.
#    An ORT that ships its own libwebgpu_dawn.so is refused: two Dawns in
#    one process cannot share a device.
# ---------------------------------------------------------------------------
set(ORT_ROOT "$ENV{HOME}/opt/onnxruntime-webgpu-extdawn" CACHE PATH
    "onnxruntime install built with the in-tree WebGPU EP and USE_EXTERNAL_DAWN")
set(DAWN_ROOT "$ENV{HOME}/opt/dawn" CACHE PATH
    "standalone Dawn install (monolithic shared libwebgpu_dawn.so)")
# find_* results are cache variables and would survive a change of
# ORT_ROOT / DAWN_ROOT (pointing at the previous install); re-probe every
# configure instead.
foreach(v ORT_LIB ORT_INCLUDE DAWN_LIB ORT_OWN_DAWN)
    unset(${v} CACHE)
endforeach()
find_library(ORT_LIB onnxruntime PATHS ${ORT_ROOT}/lib64 ${ORT_ROOT}/lib NO_DEFAULT_PATH)
find_path(ORT_INCLUDE onnxruntime_c_api.h PATHS ${ORT_ROOT}/include/onnxruntime NO_DEFAULT_PATH)
find_library(DAWN_LIB webgpu_dawn PATHS ${DAWN_ROOT}/lib64 ${DAWN_ROOT}/lib NO_DEFAULT_PATH)
find_library(ORT_OWN_DAWN webgpu_dawn PATHS ${ORT_ROOT}/lib64 ${ORT_ROOT}/lib NO_DEFAULT_PATH)
if(ORT_OWN_DAWN)
    message(FATAL_ERROR "ORT_ROOT=${ORT_ROOT} ships its own libwebgpu_dawn.so; "
                        "build onnxruntime with --use_external_dawn (see the comment above)")
endif()
set(DAWN_INCLUDE ${DAWN_ROOT}/include)

if(ORT_LIB AND DAWN_LIB AND ORT_INCLUDE AND (GLSLC OR GLSLANG))
    get_filename_component(ORT_LIBDIR ${ORT_LIB} DIRECTORY)
    get_filename_component(DAWN_LIBDIR ${DAWN_LIB} DIRECTORY)
    message(STATUS "infer: ORT ${ORT_LIB}; Dawn ${DAWN_LIB}")
    add_spirv_shaders(INFER_SPV infer_gen.comp infer_gen_img.comp)
    # runtime-compiled shaders (GLES, WGSL) as C strings
    embed_glsl(${CMAKE_CURRENT_BINARY_DIR}/infer_hash_glsl.h
               ${CMAKE_CURRENT_SOURCE_DIR}/shaders/infer_hash.glsl infer_hash_glsl)
    embed_glsl(${CMAKE_CURRENT_BINARY_DIR}/infer_gen_es_glsl.h
               ${CMAKE_CURRENT_SOURCE_DIR}/shaders/infer_gen_es.frag infer_gen_es_glsl)
    embed_glsl(${CMAKE_CURRENT_BINARY_DIR}/infer_gen_wgsl.h
               ${CMAKE_CURRENT_SOURCE_DIR}/shaders/infer_gen.wgsl infer_gen_wgsl)
    embed_glsl(${CMAKE_CURRENT_BINARY_DIR}/infer_relayout_wgsl.h
               ${CMAKE_CURRENT_SOURCE_DIR}/shaders/infer_relayout.wgsl infer_relayout_wgsl)

    add_library(infer STATIC infer/infer.h infer/infer_util.h infer/infer_tensor.c
                             infer/infer_ctx_gl.c infer/infer_ctx_vk.c infer/infer_ctx_wgpu.c
                             infer/infer_edge.c infer/infer_engine_ort.c ${INFER_SPV})
    target_include_directories(infer PUBLIC infer ${ORT_INCLUDE} ${DAWN_INCLUDE}
                                     PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(infer PRIVATE SHADER_DIR="${SHADER_OUT_DIR}" INFER_ORT_EXTERNAL_DAWN=1)
    target_link_libraries(infer PUBLIC ${ORT_LIB} ${DAWN_LIB}
                                       PkgConfig::GLES_DMABUF_STACK PkgConfig::VULKAN
                                       PkgConfig::LIBDRM m ${CMAKE_DL_LIBS})

    add_executable(hello_inference hello/hello_inference.c)
    target_link_libraries(hello_inference PRIVATE infer)
    set_target_properties(hello_inference PROPERTIES BUILD_RPATH "${ORT_LIBDIR};${DAWN_LIBDIR}")

    enable_testing()
    add_test(NAME infer_cpu_cpu
             COMMAND hello_inference --src cpu --ep cpu --iters 3 --strict
                     --model ${CMAKE_CURRENT_SOURCE_DIR}/models/synthetic_conv.onnx)
    add_test(NAME infer_all_synthetic
             COMMAND hello_inference --all --iters 5 --strict
                     --model ${CMAKE_CURRENT_SOURCE_DIR}/models/synthetic_conv.onnx)
    set_tests_properties(infer_all_synthetic PROPERTIES LABELS gpu)
else()
    message(WARNING "onnxruntime with WebGPU not found under ORT_ROOT=${ORT_ROOT} "
                    "(or no GLSL compiler): skipping infer / hello_inference")
endif()
