/* hand_overlay_vk.h -- the shared overlay triangles, Vulkan side.
 *
 * One pipeline with NO descriptor set (the only inputs are a vertex
 * buffer and 16 bytes of push constant), and one host-visible vertex
 * buffer per lane, rewritten each frame. The pipeline is created with a
 * VkPipelineRenderingCreateInfo on the presenter's colour format, so the
 * same object serves both the window pass and the camera-sized scratch
 * pass -- the trick scene_cam_vk.c's create_pipeline already uses.
 */
#ifndef HELLO_WAYLAND_HAND_OVERLAY_VK_H
#define HELLO_WAYLAND_HAND_OVERLAY_VK_H

#include "hand_overlay.h"
#include "vk_present.h"

struct hand_overlay_vk {
    VkPipelineLayout layout;
    VkPipeline pipeline;
    /* one per lane: the render thread writes lane L's buffer while the
     * GPU may still be reading lane 1-L's, the same reason the camera
     * scene keeps per-lane staging */
    VkBuffer vbo[VK_PRESENT_LANES];
    VkDeviceMemory mem[VK_PRESENT_LANES];
    void* mapped[VK_PRESENT_LANES];
    int capacity; /* vertices per lane */
};

void hand_overlay_vk_init(struct hand_overlay_vk* g, struct vk_presenter* p);
void hand_overlay_vk_fini(struct hand_overlay_vk* g, struct vk_presenter* p);

/* Record the draw into `fr`'s command buffer. Must be inside a rendering
 * instance whose colour attachment matches the presenter's format.
 * (sx, sy, ox, oy) is frame-normalised [0,1]^2 -> NDC, exactly as
 * hand_overlay_gles_rect computes it -- and Vulkan's NDC has y already
 * pointing DOWN, which is why the camera scene flips its projection
 * rather than its data. */
void hand_overlay_vk_draw(struct hand_overlay_vk* g,
                          const struct vk_frame* fr,
                          const struct hand_vert* v,
                          int n,
                          float sx,
                          float sy,
                          float ox,
                          float oy);

#endif
