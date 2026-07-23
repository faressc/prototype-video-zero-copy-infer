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
# 2. Generate the xdg-shell binding.
#    xdg-shell is NOT part of core Wayland. Core only knows wl_surface.
#    "How a surface becomes a desktop window" lives in this extension,
#    shipped as an XML spec. wayland-scanner generates:
#      - a header  (the request wrappers + listener structs)
#      - a .c file (the interface descriptor tables, like wl_surface_interface)
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

# ---------------------------------------------------------------------------
# 3. Build the executable from our code + the generated code.
# ---------------------------------------------------------------------------
add_executable(hello_wayland
    main.c
    ${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-protocol.c
    ${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-client-protocol.h)

# So that #include "xdg-shell-client-protocol.h" finds the generated header.
target_include_directories(hello_wayland PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(hello_wayland PRIVATE PkgConfig::WAYLAND_CLIENT PkgConfig::XKBCOMMON
                                            PkgConfig::WAYLAND_CURSOR)
