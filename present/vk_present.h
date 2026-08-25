/* vk_present.h -- the Vulkan "presenter": everything between the
 * Wayland layer (common.c) and the scene that records draw commands.
 *
 * Extracted from vulkan.c, which stays as the untouched single-file
 * reference (README §16 explains every object in here). The presenter
 * owns: instance, surface, device + the one queue, swapchain + views,
 * the optional depth image, the command pool, and all three kinds of
 * sync object -- fences and image_avail per LANE (frame in flight),
 * render_done per IMAGE.
 *
 * The seam is `record(frame)`: the presenter has recorded the
 * to-color barrier (and the depth barrier), the scene records
 * vkCmdBeginRendering ... vkCmdEndRendering, the presenter records the
 * to-present barrier, submits, and presents. The scene owns
 * vkCmdBeginRendering because it decides the attachments (clear
 * values, whether depth is used).
 *
 * Clip space is Vulkan's: y DOWN, z in [0,1]. Build projections with
 * mat4_perspective(..., y_down = 1, z_zero_to_one = 1).
 */
#ifndef HELLO_WAYLAND_VK_PRESENT_H
#define HELLO_WAYLAND_VK_PRESENT_H

#include <time.h>

#include "common.h"

#define VK_USE_PLATFORM_WAYLAND_KHR
#include <vulkan/vulkan.h>

enum {
    VK_PRESENT_DEPTH = 1 << 0, /* create a depth image; frames carry a depth view */
    /* enable what importing camera dmabufs needs (external memory fd +
     * dma_buf, DRM format modifiers, foreign queue family, sampler
     * YCbCr conversion) when the driver has it; see dmabuf_import */
    VK_PRESENT_CAMERA_IMPORT = 1 << 1,
};

enum { VK_PRESENT_LANES = 2 }; /* MAX_FRAMES_IN_FLIGHT */
enum { VK_PRESENT_MAX_IMAGES = 8 };

/* Handed to the scene between the presenter's two barriers. */
struct vk_frame {
    VkCommandBuffer cmd; /* recording */
    VkImage color_image;
    VkImageView color;    /* in COLOR_ATTACHMENT_OPTIMAL */
    VkImageView depth;    /* VK_NULL_HANDLE without VK_PRESENT_DEPTH; else in
                           * DEPTH_STENCIL_ATTACHMENT_OPTIMAL, contents undefined */
    uint32_t image_index; /* swapchain image, for per-image scene resources */
    uint32_t lane;        /* frame in flight, for per-lane scene resources */
    uint64_t serial;      /* this frame's number; vk_present_serial_done() later */
    VkExtent2D extent;
};

struct vk_presenter;

struct vk_scene {
    /* Device and swapchain exist (color_format, depth_format known).
     * Build pipelines, upload meshes. Called once. */
    void (*init)(struct vk_presenter* p, void* user);
    /* Optional: the swapchain was (re)created at this extent. */
    void (*resize)(struct vk_presenter* p, VkExtent2D extent, void* user);
    /* Record one frame into frame->cmd. */
    void (*record)(struct vk_presenter* p, const struct vk_frame* frame, void* user);
    /* Device idle. Destroy what init created. */
    void (*fini)(struct vk_presenter* p, void* user);
};

struct vk_presenter {
    struct app app; /* MUST be first */

    unsigned flags;
    const struct vk_scene* scene;
    void* user;
    int resized;

    VkInstance instance;
    VkSurfaceKHR surface;
    VkPhysicalDevice phys;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t queue_family;
    VkDevice device;
    VkQueue queue;
    int dmabuf_import; /* VK_PRESENT_CAMERA_IMPORT requested and fully available */

    VkSwapchainKHR swapchain;
    VkFormat color_format;
    VkExtent2D extent;
    uint32_t image_count;
    VkImage images[VK_PRESENT_MAX_IMAGES];
    VkImageView views[VK_PRESENT_MAX_IMAGES];
    VkSemaphore render_done[VK_PRESENT_MAX_IMAGES]; /* per image */

    /* depth: one image shared by all lanes; a barrier at the start of
     * each frame orders it against the previous frame's use */
    VkFormat depth_format;
    VkImageAspectFlags depth_aspect;
    VkImage depth_image;
    VkDeviceMemory depth_memory;
    VkImageView depth_view;

    VkCommandPool cmd_pool;
    VkCommandBuffer cmd[VK_PRESENT_LANES];
    VkSemaphore image_avail[VK_PRESENT_LANES]; /* per lane */
    VkFence in_flight[VK_PRESENT_LANES];       /* per lane */
    uint32_t lane;
    uint64_t frame_serial;                  /* counts submitted frames */
    uint64_t lane_serial[VK_PRESENT_LANES]; /* which frame each lane carries */

    struct timespec t0;
};

/* Connect (app_init), wait for the first configure, bring up Vulkan,
 * create the swapchain (+ depth), then scene->init. Returns 0 on
 * success. */
int vk_present_init(struct vk_presenter* p,
                    unsigned flags,
                    const struct vk_scene* scene,
                    void* user,
                    const char* title,
                    const char* app_id);

/* The game-style loop: dispatch_pending, flush, clock, one frame;
 * FIFO present paces it to the display. */
void vk_present_run(struct vk_presenter* p);

/* vkDeviceWaitIdle, scene->fini, then tear everything down. */
void vk_present_fini(struct vk_presenter* p);

/* Helpers scenes need. */
uint32_t vk_present_find_memory_type(const struct vk_presenter* p,
                                     uint32_t type_bits,
                                     VkMemoryPropertyFlags required);
VkShaderModule vk_present_load_shader(const struct vk_presenter* p, const char* path);

/* Has the GPU finished the frame with this serial (from vk_frame)?
 * The scene's way to know a resource read by that frame -- a camera
 * buffer -- may be handed back to its owner. */
int vk_present_serial_done(const struct vk_presenter* p, uint64_t serial);

#define VK_CHECK(expr)                                                  \
    do {                                                                \
        VkResult vk_check_r = (expr);                                   \
        if (vk_check_r != VK_SUCCESS) {                                 \
            fprintf(stderr, "%s failed: %d\n", #expr, (int)vk_check_r); \
            exit(1);                                                    \
        }                                                               \
    } while (0)

#endif
