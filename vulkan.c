/* vulkan.c -- the explicit GPU backend.
 *
 * Where EGL hid the machinery, Vulkan hands you the levers -- each one
 * a shm.c concept wearing an API name:
 *
 *   minImageCount / present mode  = SLOTS / double-vs-triple
 *   vkAcquireNextImageKHR         = the "find a !busy slot" scan
 *   imageAvailable semaphore      = busy -> 0 (release arrived)
 *   vkQueuePresentKHR             = attach + damage + commit
 *   swapchain recreation          = fresh-pool-per-resize
 *                                   (oldSwapchain = the orphan pattern)
 *
 * Shaders are compiled AT BUILD TIME (CMake runs glslc: GLSL -> SPIR-V,
 * the wayland-scanner pattern: spec compiled to artifact); the driver
 * only lowers SPIR-V -> GPU ISA at pipeline creation.
 *
 * Loop style: redraw hook = NULL, so common.c runs no frame loop --
 * this backend owns its loop: blocking FIFO present (game style),
 * servicing protocol events with dispatch_pending. Mesa's WSI answers
 * wl_surface.frame callbacks internally to throttle presents.
 *
 * Uses Vulkan 1.3 dynamic rendering (no render passes / framebuffers);
 * Honeykrisp (Asahi) qualifies.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"

#define VK_USE_PLATFORM_WAYLAND_KHR
#include <vulkan/vulkan.h>

#ifndef SHADER_DIR
#define SHADER_DIR "./shaders"
#endif

enum { MAX_FRAMES_IN_FLIGHT = 2 }; /* CPU may record this many frames ahead */
enum { MAX_SWAPCHAIN_IMAGES = 8 };

struct push_constants { /* must match the push_constant block in gradient.frag */
    float size[2];
    float cursor[2];
    float time_ms;
};

struct vk_app {
    struct app app; /* MUST be first */

    int resized; /* configure changed the size -> recreate swapchain */

    VkInstance instance;
    VkSurfaceKHR vk_surface;
    VkPhysicalDevice phys;
    uint32_t queue_family;
    VkDevice device;
    VkQueue queue;

    VkSwapchainKHR swapchain;
    VkFormat format;
    VkExtent2D extent;
    uint32_t image_count;
    VkImage images[MAX_SWAPCHAIN_IMAGES];
    VkImageView views[MAX_SWAPCHAIN_IMAGES];
    /* renderFinished is per swapchain IMAGE (present may still read an
     * image's semaphore while we start the next frame) ... */
    VkSemaphore render_done[MAX_SWAPCHAIN_IMAGES];

    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkCommandPool cmd_pool;

    /* ... while acquire/fence sync is per FRAME IN FLIGHT. */
    VkCommandBuffer cmd[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore image_avail[MAX_FRAMES_IN_FLIGHT];
    VkFence in_flight[MAX_FRAMES_IN_FLIGHT];
    uint32_t frame;

    struct timespec t0;
};

static void die(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

#define VK_CHECK(expr)                                                  \
    do {                                                                \
        VkResult vk_check_r = (expr);                                   \
        if (vk_check_r != VK_SUCCESS) {                                 \
            fprintf(stderr, "%s failed: %d\n", #expr, (int)vk_check_r); \
            exit(1);                                                    \
        }                                                               \
    } while (0)

static uint32_t now_ms(struct vk_app* v) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint32_t)((t.tv_sec - v->t0.tv_sec) * 1000 + (t.tv_nsec - v->t0.tv_nsec) / 1000000);
}

/* The backend hook: the loop draws, so configure only records that the
 * swapchain is stale -- Vulkan's official resize story IS our
 * fresh-pool-per-resize. */
static void vk_configure(struct app* a) {
    struct vk_app* v = (struct vk_app*)a;
    if (v->swapchain &&
        ((uint32_t)a->width != v->extent.width || (uint32_t)a->height != v->extent.height)) {
        v->resized = 1;
    }
}

static const struct app_backend vk_backend = {
    .configure = vk_configure,
    .redraw = NULL, /* we own the loop */
};

/* ------------------------------------------------------------------ */
/* Vulkan bring-up                                                    */
/* ------------------------------------------------------------------ */

static void create_instance(struct vk_app* v) {
    const char* exts[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
    };
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "hello-wayland-vulkan",
        .apiVersion = VK_API_VERSION_1_3, /* dynamic rendering is core here */
    };
    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = exts,
    };
    VK_CHECK(vkCreateInstance(&ci, NULL, &v->instance));
}

/* Presentation support is per (device, queue family, display) triple,
 * hence the dedicated Wayland query. */
static void pick_device(struct vk_app* v) {
    VkPhysicalDevice devs[8];
    uint32_t n = 8;
    /* VK_INCOMPLETE (a positive STATUS, not an error) = "array too
     * small, results truncated" -- fine, we only need one device. */
    VkResult r = vkEnumeratePhysicalDevices(v->instance, &n, devs);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) { die("vkEnumeratePhysicalDevices failed"); }
    if (n == 0) { die("no Vulkan device"); }

    for (uint32_t d = 0; d < n; d++) {
        VkQueueFamilyProperties fams[16];
        uint32_t fn = 16;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &fn, fams);
        for (uint32_t f = 0; f < fn; f++) {
            if ((fams[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                vkGetPhysicalDeviceWaylandPresentationSupportKHR(devs[d], f, v->app.display)) {
                v->phys = devs[d];
                v->queue_family = f;

                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(v->phys, &props);
                fprintf(stderr, "using GPU: %s\n", props.deviceName);
                return;
            }
        }
    }
    die("no queue family with graphics + Wayland present support");
}

static void create_device(struct vk_app* v) {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = v->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    VkPhysicalDeviceDynamicRenderingFeatures dynren = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
        .dynamicRendering = VK_TRUE,
    };
    const char* exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &dynren,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = exts,
    };
    VK_CHECK(vkCreateDevice(v->phys, &ci, NULL, &v->device));
    vkGetDeviceQueue(v->device, v->queue_family, 0, &v->queue);
}

/* The swapchain: SLOTS as a driver object. On Wayland the WSI
 * allocates dmabufs, wraps them in wl_buffers, and tracks
 * wl_buffer.release for us. */
static void create_swapchain(struct vk_app* v) {
    struct app* a = &v->app;

    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(v->phys, v->vk_surface, &caps));

    /* 0xFFFFFFFF = "no fixed size, you pick" -- the Vulkan spelling of
     * the configure event's width=0. Normal case on Wayland. */
    if (caps.currentExtent.width != 0xFFFFFFFFu) {
        v->extent = caps.currentExtent;
    } else {
        v->extent.width = (uint32_t)a->width;
        v->extent.height = (uint32_t)a->height;
        if (v->extent.width < caps.minImageExtent.width) {
            v->extent.width = caps.minImageExtent.width;
        }
        if (v->extent.height < caps.minImageExtent.height) {
            v->extent.height = caps.minImageExtent.height;
        }
        if (v->extent.width > caps.maxImageExtent.width) {
            v->extent.width = caps.maxImageExtent.width;
        }
        if (v->extent.height > caps.maxImageExtent.height) {
            v->extent.height = caps.maxImageExtent.height;
        }
    }

    /* Honeykrisp advertises 32+ format/colorspace combos, so this
     * query legitimately returns VK_INCOMPLETE (truncated) -- harmless,
     * we pick one format from whatever fits. */
    VkSurfaceFormatKHR formats[32];
    uint32_t fn = 32;
    VkResult fr = vkGetPhysicalDeviceSurfaceFormatsKHR(v->phys, v->vk_surface, &fn, formats);
    if (fr != VK_SUCCESS && fr != VK_INCOMPLETE) {
        die("vkGetPhysicalDeviceSurfaceFormatsKHR failed");
    }
    VkSurfaceFormatKHR pick = formats[0];
    for (uint32_t i = 0; i < fn; i++) {
        if (formats[i].format == VK_FORMAT_B8G8R8A8_UNORM) { pick = formats[i]; }
    }
    v->format = pick.format;

    uint32_t count = caps.minImageCount;
    if (caps.maxImageCount > 0 && count > caps.maxImageCount) { count = caps.maxImageCount; }

    VkSwapchainKHR old = v->swapchain;
    VkSwapchainCreateInfoKHR ci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = v->vk_surface,
        .minImageCount = count,
        .imageFormat = pick.format,
        .imageColorSpace = pick.colorSpace,
        .imageExtent = v->extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,
        .clipped = VK_TRUE,
        .oldSwapchain = old, /* the orphan pattern: on-loan images drain
                              * out of the old chain */
    };
    VK_CHECK(vkCreateSwapchainKHR(v->device, &ci, NULL, &v->swapchain));
    if (old) { vkDestroySwapchainKHR(v->device, old, NULL); }

    v->image_count = MAX_SWAPCHAIN_IMAGES;
    VK_CHECK(vkGetSwapchainImagesKHR(v->device, v->swapchain, &v->image_count, v->images));

    for (uint32_t i = 0; i < v->image_count; i++) {
        VkImageViewCreateInfo vci = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = v->images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = v->format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        VK_CHECK(vkCreateImageView(v->device, &vci, NULL, &v->views[i]));

        VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(v->device, &sci, NULL, &v->render_done[i]));
    }
}

static void destroy_swapchain_views(struct vk_app* v) {
    for (uint32_t i = 0; i < v->image_count; i++) {
        vkDestroyImageView(v->device, v->views[i], NULL);
        vkDestroySemaphore(v->device, v->render_done[i], NULL);
    }
}

static void recreate_swapchain(struct vk_app* v) {
    vkDeviceWaitIdle(v->device); /* simple & correct; real engines overlap */
    destroy_swapchain_views(v);
    create_swapchain(v); /* passes oldSwapchain internally */
    v->resized = 0;
}

static VkShaderModule load_shader(struct vk_app* v, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* code = malloc((size_t)size);
    if (fread(code, 1, (size_t)size, f) != (size_t)size) { die("shader read failed"); }
    fclose(f);

    VkShaderModuleCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)size,
        .pCode = code, /* SPIR-V words, straight from the build step */
    };
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(v->device, &ci, NULL, &mod));
    free(code);
    return mod;
}

/* Pipeline creation is where the driver does its half of the shader
 * compile: SPIR-V -> NIR -> AGX. The expensive step games hide behind
 * loading screens; Mesa caches it on disk. */
static void create_pipeline(struct vk_app* v) {
    VkShaderModule vs = load_shader(v, SHADER_DIR "/gradient.vert.spv");
    VkShaderModule fs = load_shader(v, SHADER_DIR "/gradient.frag.spv");

    VkPushConstantRange pc_range = {
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(struct push_constants),
    };
    VkPipelineLayoutCreateInfo lci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc_range,
    };
    VK_CHECK(vkCreatePipelineLayout(v->device, &lci, NULL, &v->pipeline_layout));

    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vs,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fs,
         .pName = "main"},
    };

    /* No vertex buffers: gl_VertexIndex generates the triangle. */
    VkPipelineVertexInputStateCreateInfo vin = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState blend_att = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    VkPipelineColorBlendStateCreateInfo blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blend_att};

    /* viewport/scissor dynamic: a resize rebuilds only the swapchain,
     * never the pipeline. */
    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dyn_states};

    /* Dynamic rendering: declare the attachment format instead of
     * building render pass + framebuffer objects. */
    VkPipelineRenderingCreateInfo rendering = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &v->format};

    VkGraphicsPipelineCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vin,
        .pInputAssemblyState = &ia,
        .pViewportState = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pColorBlendState = &blend,
        .pDynamicState = &dyn,
        .layout = v->pipeline_layout,
    };
    VK_CHECK(vkCreateGraphicsPipelines(v->device, VK_NULL_HANDLE, 1, &pci, NULL, &v->pipeline));

    vkDestroyShaderModule(v->device, vs, NULL);
    vkDestroyShaderModule(v->device, fs, NULL);
}

static void create_sync_and_commands(struct vk_app* v) {
    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = v->queue_family,
    };
    VK_CHECK(vkCreateCommandPool(v->device, &pci, NULL, &v->cmd_pool));

    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = v->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };
    VK_CHECK(vkAllocateCommandBuffers(v->device, &cai, v->cmd));

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(v->device, &sci, NULL, &v->image_avail[i]));
        /* fences start SIGNALED so frame 0 doesn't wait forever */
        VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                 .flags = VK_FENCE_CREATE_SIGNALED_BIT};
        VK_CHECK(vkCreateFence(v->device, &fci, NULL, &v->in_flight[i]));
    }
}

/* ------------------------------------------------------------------ */
/* One frame -- every step is a shm.c concept made explicit:          */
/*   wait fence      = don't record over a frame the GPU still runs   */
/*   acquire         = find a released slot (blocks if none, FIFO)    */
/*   record + submit = redraw -- but into a command buffer            */
/*   present         = attach + damage + commit                       */
/* ------------------------------------------------------------------ */

static void draw_frame(struct vk_app* v) {
    struct app* a = &v->app;
    uint32_t f = v->frame;

    VK_CHECK(vkWaitForFences(v->device, 1, &v->in_flight[f], VK_TRUE, UINT64_MAX));

    uint32_t img;
    VkResult r = vkAcquireNextImageKHR(v->device,
                                       v->swapchain,
                                       UINT64_MAX,
                                       v->image_avail[f],
                                       VK_NULL_HANDLE,
                                       &img);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || v->resized) {
        recreate_swapchain(v);
        return; /* image_avail[f] unsignaled -- safe to reuse next call */
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) { die("acquire failed"); }

    VK_CHECK(vkResetFences(v->device, 1, &v->in_flight[f]));

    VkCommandBuffer cmd = v->cmd[f];
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    /* Dynamic rendering means WE manage image layouts. */
    VkImageMemoryBarrier to_color = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = v->images[img],
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0,
                         0,
                         NULL,
                         0,
                         NULL,
                         1,
                         &to_color);

    VkRenderingAttachmentInfo color_att = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = v->views[img],
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, /* we overwrite every pixel */
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingInfo ri = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, v->extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_att,
    };
    vkCmdBeginRendering(cmd, &ri);

    VkViewport viewport = {0, 0, (float)v->extent.width, (float)v->extent.height, 0.0f, 1.0f};
    VkRect2D scissor = {{0, 0}, v->extent};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v->pipeline);

    struct push_constants pc = {
        .size = {(float)v->extent.width, (float)v->extent.height},
        .cursor = {(float)a->ptr_x, (float)a->ptr_y},
        .time_ms = (float)app_anim_time(a),
    };
    vkCmdPushConstants(cmd, v->pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(cmd, 3, 1, 0, 0); /* the fullscreen triangle */

    vkCmdEndRendering(cmd);

    VkImageMemoryBarrier to_present = to_color;
    to_present.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    to_present.dstAccessMask = 0;
    to_present.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0,
                         0,
                         NULL,
                         0,
                         NULL,
                         1,
                         &to_present);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &v->image_avail[f], /* GPU waits for the slot */
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &v->render_done[img],
    };
    VK_CHECK(vkQueueSubmit(v->queue, 1, &si, v->in_flight[f]));

    VkPresentInfoKHR pi = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &v->render_done[img], /* present waits for the GPU */
        .swapchainCount = 1,
        .pSwapchains = &v->swapchain,
        .pImageIndices = &img,
    };
    r = vkQueuePresentKHR(v->queue, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || v->resized) {
        recreate_swapchain(v);
    } else if (r != VK_SUCCESS) {
        die("present failed");
    }

    v->frame = (f + 1) % MAX_FRAMES_IN_FLIGHT;
}

/* ------------------------------------------------------------------ */

int main(void) {
    struct vk_app v = {0};
    clock_gettime(CLOCK_MONOTONIC, &v.t0);

    if (app_init(&v.app, &vk_backend, "hello wayland (Vulkan)", "hello-wayland-vulkan") < 0) {
        return 1;
    }

    /* Wait for the configure/ack handshake: only after acking may a
     * buffer appear, and Vulkan's first present is that first buffer. */
    while (!v.app.configured && wl_display_dispatch(v.app.display) != -1) {}

    create_instance(&v);
    VkWaylandSurfaceCreateInfoKHR sci = {
        .sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
        .display = v.app.display,
        .surface = v.app.surface, /* the bridge: wl_surface -> VkSurfaceKHR */
    };
    VK_CHECK(vkCreateWaylandSurfaceKHR(v.instance, &sci, NULL, &v.vk_surface));
    pick_device(&v);
    create_device(&v);
    create_swapchain(&v);
    create_pipeline(&v);
    create_sync_and_commands(&v);

    /* Game-style loop: service protocol events non-blocking, then let
     * FIFO present throttle us to the display. */
    while (v.app.running) {
        if (wl_display_dispatch_pending(v.app.display) == -1) { break; }
        wl_display_flush(v.app.display);
        app_advance_clock(&v.app, now_ms(&v)); /* no frame callbacks here */
        draw_frame(&v);
    }

    vkDeviceWaitIdle(v.device);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(v.device, v.image_avail[i], NULL);
        vkDestroyFence(v.device, v.in_flight[i], NULL);
    }
    destroy_swapchain_views(&v);
    vkDestroySwapchainKHR(v.device, v.swapchain, NULL);
    vkDestroyPipeline(v.device, v.pipeline, NULL);
    vkDestroyPipelineLayout(v.device, v.pipeline_layout, NULL);
    vkDestroyCommandPool(v.device, v.cmd_pool, NULL);
    vkDestroyDevice(v.device, NULL);
    vkDestroySurfaceKHR(v.instance, v.vk_surface, NULL);
    vkDestroyInstance(v.instance, NULL);

    app_finish(&v.app);
    return 0;
}
