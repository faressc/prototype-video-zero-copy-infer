cmake_minimum_required(VERSION 3.20)
project(hello_wayland C)

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
# 2. Generate the xdg-shell binding (shared by all three executables).
# ---------------------------------------------------------------------------
set(XDG_SHELL_XML ${WAYLAND_PROTOCOLS_DIR}/stable/xdg-shell/xdg-shell.xml)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-client-protocol.h
    COMMAND ${WAYLAND_SCANNER} client-header ${XDG_SHELL_XML}
            ${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-client-protocol.h
    DEPENDS ${XDG_SHELL_XML})

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-protocol.c
    COMMAND ${WAYLAND_SCANNER} private-code ${XDG_SHELL_XML}
            ${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-protocol.c
    DEPENDS ${XDG_SHELL_XML})

add_library(xdg_shell_protocol STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-protocol.c
    ${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-client-protocol.h)
target_include_directories(xdg_shell_protocol PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(xdg_shell_protocol PUBLIC PkgConfig::WAYLAND_CLIENT)

# ---------------------------------------------------------------------------
# 3. app_common -- the shared Wayland client layer (registry, xdg role +
#    configure/ack, seat input with xkbcommon, themed cursor, animation
#    clock, frame-callback loop). Backends provide two hooks.
# ---------------------------------------------------------------------------
add_library(app_common STATIC common.c common.h)
target_link_libraries(app_common PUBLIC xdg_shell_protocol
                                        PkgConfig::XKBCOMMON
                                        PkgConfig::WAYLAND_CURSOR
                                        m) # fmod in the animation clock

# ---------------------------------------------------------------------------
# 4. hello_wayland -- CPU rendering into wl_shm buffers.
# ---------------------------------------------------------------------------
add_executable(hello_wayland shm.c)
target_link_libraries(hello_wayland PRIVATE app_common)

# ---------------------------------------------------------------------------
# 5. hello_wayland_gles_dmabuf -- raw GLES + a hand-rolled dmabuf
#    swapchain. EGL is kept ONLY as context factory (surfaceless); its
#    window-system half (EGLSurface/eglSwapBuffers) is replaced by
#    gbm_bo allocation + the linux-dmabuf protocol, bound by US -- so
#    this needs its own generated protocol binding, like xdg-shell.
#    GLSL still compiles AT RUNTIME inside the driver.
# ---------------------------------------------------------------------------
set(DMABUF_XML ${WAYLAND_PROTOCOLS_DIR}/stable/linux-dmabuf/linux-dmabuf-v1.xml)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/linux-dmabuf-v1-client-protocol.h
    COMMAND ${WAYLAND_SCANNER} client-header ${DMABUF_XML}
            ${CMAKE_CURRENT_BINARY_DIR}/linux-dmabuf-v1-client-protocol.h
    DEPENDS ${DMABUF_XML})

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/linux-dmabuf-v1-protocol.c
    COMMAND ${WAYLAND_SCANNER} private-code ${DMABUF_XML}
            ${CMAKE_CURRENT_BINARY_DIR}/linux-dmabuf-v1-protocol.c
    DEPENDS ${DMABUF_XML})

add_library(dmabuf_protocol STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/linux-dmabuf-v1-protocol.c
    ${CMAKE_CURRENT_BINARY_DIR}/linux-dmabuf-v1-client-protocol.h)
target_include_directories(dmabuf_protocol PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(dmabuf_protocol PUBLIC PkgConfig::WAYLAND_CLIENT)

pkg_check_modules(GLES_DMABUF_STACK REQUIRED IMPORTED_TARGET egl glesv2 gbm libdrm)
add_executable(hello_wayland_gles_dmabuf gles-dmabuf.c)
target_link_libraries(hello_wayland_gles_dmabuf PRIVATE app_common dmabuf_protocol
                                                        PkgConfig::GLES_DMABUF_STACK)

# ---------------------------------------------------------------------------
# 5b. hello_wayland_gles_eglsurface -- the same GLES gradient through
#     the CLASSIC path: wl_egl_window + EGLSurface + eglSwapBuffers,
#     swapchain hidden inside Mesa (how GTK/Qt/weston-simple-egl work).
#     Diff against gles-dmabuf.c: the delta IS eglSwapBuffers.
# ---------------------------------------------------------------------------
pkg_check_modules(GLES_EGLSURFACE_STACK REQUIRED IMPORTED_TARGET egl glesv2 wayland-egl)
add_executable(hello_wayland_gles_eglsurface gles-eglsurface.c)
target_link_libraries(hello_wayland_gles_eglsurface PRIVATE app_common
                                                            PkgConfig::GLES_EGLSURFACE_STACK)

# ---------------------------------------------------------------------------
# 6. hello_wayland_vulkan -- GPU rendering via Vulkan.
#    Shaders compile AT BUILD TIME to SPIR-V -- the same pattern as
#    wayland-scanner: a spec (GLSL) is compiled into an artifact (.spv)
#    the runtime then merely loads. Needs glslc or glslangValidator.
# ---------------------------------------------------------------------------
pkg_check_modules(VULKAN REQUIRED IMPORTED_TARGET vulkan)
find_program(GLSLC glslc)
find_program(GLSLANG glslangValidator)

if(GLSLC OR GLSLANG)
    set(SHADER_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    file(MAKE_DIRECTORY ${SHADER_OUT_DIR})
    foreach(SHADER gradient.vert gradient.frag)
        set(SPV ${SHADER_OUT_DIR}/${SHADER}.spv)
        if(GLSLC)
            add_custom_command(
                OUTPUT ${SPV}
                COMMAND ${GLSLC} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER} -o ${SPV}
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER})
        else()
            add_custom_command(
                OUTPUT ${SPV}
                COMMAND ${GLSLANG} -V ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER} -o ${SPV}
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER})
        endif()
        list(APPEND SPV_FILES ${SPV})
    endforeach()

    add_executable(hello_wayland_vulkan vulkan.c ${SPV_FILES})
    target_compile_definitions(hello_wayland_vulkan PRIVATE
                               SHADER_DIR="${SHADER_OUT_DIR}")
    target_link_libraries(hello_wayland_vulkan PRIVATE app_common PkgConfig::VULKAN)
else()
    message(WARNING "No GLSL->SPIR-V compiler found: skipping hello_wayland_vulkan. "
                    "Install one with: sudo dnf install glslc   (or: glslang)")
endif()
