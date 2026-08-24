/* vk_present.c -- see vk_present.h. vulkan.c's bring-up and draw_frame,
 * with the gradient pipeline cut out and a scene vtable put in its
 * place. Additions: the depth image (the first vkAllocateMemory in the
 * project) and its per-frame barrier. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vk_present.h"

static void die(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static uint32_t now_ms(struct vk_presenter* p) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint32_t)((t.tv_sec - p->t0.tv_sec) * 1000 + (t.tv_nsec - p->t0.tv_nsec) / 1000000);
}

/* ------------------------------------------------------------------ */
/* common.c hook: the loop draws, configure only marks the swapchain  */
/* stale.                                                             */
/* ------------------------------------------------------------------ */

static void present_configure(struct app* a) {
    struct vk_presenter* p = (struct vk_presenter*)a;
    if (p->swapchain &&
        ((uint32_t)a->width != p->extent.width || (uint32_t)a->height != p->extent.height)) {
        p->resized = 1;
    }
}

static const struct app_backend present_backend = {
    .configure = present_configure,
    .redraw = NULL,
};

/* ------------------------------------------------------------------ */
/* Bring-up (vulkan.c)                                                */
/* ------------------------------------------------------------------ */

static void create_instance(struct vk_presenter* p) {
    const char* exts[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
    };
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "hello-wayland-vk-present",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = exts,
    };
    VK_CHECK(vkCreateInstance(&ci, NULL, &p->instance));
}

static void pick_device(struct vk_presenter* p) {
    VkPhysicalDevice devs[8];
    uint32_t n = 8;
    VkResult r = vkEnumeratePhysicalDevices(p->instance, &n, devs);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) { die("vkEnumeratePhysicalDevices failed"); }
    if (n == 0) { die("no Vulkan device"); }

    for (uint32_t d = 0; d < n; d++) {
        VkQueueFamilyProperties fams[16];
        uint32_t fn = 16;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &fn, fams);
        for (uint32_t f = 0; f < fn; f++) {
            if ((fams[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                vkGetPhysicalDeviceWaylandPresentationSupportKHR(devs[d], f, p->app.display)) {
                p->phys = devs[d];
                p->queue_family = f;
                vkGetPhysicalDeviceMemoryProperties(p->phys, &p->mem_props);

                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(p->phys, &props);
                fprintf(stderr, "using GPU: %s\n", props.deviceName);
                return;
            }
        }
    }
    die("no queue family with graphics + Wayland present support");
}

static void create_device(struct vk_presenter* p) {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = p->queue_family,
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
    VK_CHECK(vkCreateDevice(p->phys, &ci, NULL, &p->device));
    vkGetDeviceQueue(p->device, p->queue_family, 0, &p->queue);
}

uint32_t vk_present_find_memory_type(const struct vk_presenter* p,
                                     uint32_t type_bits,
                                     VkMemoryPropertyFlags required) {
    /* type_bits: from vkGet*MemoryRequirements -- bit i set = memory
     * type i is acceptable for this resource. Pick the first such type
     * that also has the properties the CPU side needs. */
    for (uint32_t i = 0; i < p->mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (p->mem_props.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    die("no suitable memory type");
    return 0;
}

VkShaderModule vk_present_load_shader(const struct vk_presenter* p, const char* path) {
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
        .pCode = code,
    };
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(p->device, &ci, NULL, &mod));
    free(code);
    return mod;
}

/* ------------------------------------------------------------------ */
/* Depth: an image WE allocate -- the swapchain images came from the  */
/* WSI, this one goes through the full create/query/allocate/bind     */
/* sequence every Vulkan resource takes.                              */
/* ------------------------------------------------------------------ */

static void pick_depth_format(struct vk_presenter* p) {
    const VkFormat candidates[] = {VK_FORMAT_D32_SFLOAT,
                                   VK_FORMAT_D32_SFLOAT_S8_UINT,
                                   VK_FORMAT_D24_UNORM_S8_UINT};
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(p->phys, candidates[i], &fp);
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            p->depth_format = candidates[i];
            /* a layout transition must name EVERY aspect the format has */
            p->depth_aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (candidates[i] != VK_FORMAT_D32_SFLOAT) {
                p->depth_aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
            return;
        }
    }
    die("no depth format");
}

static void create_depth(struct vk_presenter* p) {
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = p->depth_format,
        .extent = {p->extent.width, p->extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL, /* driver-chosen layout: never CPU-read */
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VK_CHECK(vkCreateImage(p->device, &ici, NULL, &p->depth_image));

    /* An image is only a description until memory is bound to it. */
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(p->device, p->depth_image, &req);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex =
            vk_present_find_memory_type(p, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    VK_CHECK(vkAllocateMemory(p->device, &mai, NULL, &p->depth_memory));
    VK_CHECK(vkBindImageMemory(p->device, p->depth_image, p->depth_memory, 0));

    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = p->depth_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = p->depth_format,
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
    };
    VK_CHECK(vkCreateImageView(p->device, &vci, NULL, &p->depth_view));
}

static void destroy_depth(struct vk_presenter* p) {
    if (p->depth_view) { vkDestroyImageView(p->device, p->depth_view, NULL); }
    if (p->depth_image) { vkDestroyImage(p->device, p->depth_image, NULL); }
    if (p->depth_memory) { vkFreeMemory(p->device, p->depth_memory, NULL); }
    p->depth_view = VK_NULL_HANDLE;
    p->depth_image = VK_NULL_HANDLE;
    p->depth_memory = VK_NULL_HANDLE;
}

/* ------------------------------------------------------------------ */
/* Swapchain (vulkan.c)                                               */
/* ------------------------------------------------------------------ */

static void create_swapchain(struct vk_presenter* p) {
    struct app* a = &p->app;

    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(p->phys, p->surface, &caps));

    if (caps.currentExtent.width != 0xFFFFFFFFu) {
        p->extent = caps.currentExtent;
    } else {
        p->extent.width = (uint32_t)a->width;
        p->extent.height = (uint32_t)a->height;
        if (p->extent.width < caps.minImageExtent.width) {
            p->extent.width = caps.minImageExtent.width;
        }
        if (p->extent.height < caps.minImageExtent.height) {
            p->extent.height = caps.minImageExtent.height;
        }
        if (p->extent.width > caps.maxImageExtent.width) {
            p->extent.width = caps.maxImageExtent.width;
        }
        if (p->extent.height > caps.maxImageExtent.height) {
            p->extent.height = caps.maxImageExtent.height;
        }
    }

    /* The two-call enumeration pattern, done properly: ask for the
     * count, then fetch ALL entries. (vulkan.c's fixed 32-entry array
     * truncates -- Honeykrisp lists more than that -- and if the UNORM
     * entry falls past the cut, formats[0] wins, which is an *_SRGB
     * format: the driver then gamma-encodes every write, and a linear
     * 0.08 clear shows up as ~0.31 gray. The gradient never noticed;
     * the cube's clear color did.) */
    uint32_t fn = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(p->phys, p->surface, &fn, NULL));
    if (fn == 0) { die("no surface formats"); }
    VkSurfaceFormatKHR* formats = malloc(fn * sizeof(*formats));
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(p->phys, p->surface, &fn, formats));

    /* UNORM = "store what the shader writes, no conversion" -- the same
     * contract as the GL paths' XRGB8888 dmabufs. An *_SRGB swapchain
     * would be the right choice for a linear-light renderer; this
     * project writes display-referred values like the GL backends do. */
    VkSurfaceFormatKHR pick = formats[0];
    int found = 0;
    for (uint32_t i = 0; i < fn && !found; i++) {
        if ((formats[i].format == VK_FORMAT_B8G8R8A8_UNORM ||
             formats[i].format == VK_FORMAT_R8G8B8A8_UNORM) &&
            formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            pick = formats[i];
            found = 1;
        }
    }
    free(formats);
    if (!found) { fprintf(stderr, "warning: no 8-bit UNORM surface format\n"); }
    if (!p->swapchain) {
        fprintf(stderr,
                "surface formats: %u, using VkFormat %d (%s)\n",
                fn,
                (int)pick.format,
                pick.format == VK_FORMAT_B8G8R8A8_UNORM   ? "B8G8R8A8_UNORM"
                : pick.format == VK_FORMAT_R8G8B8A8_UNORM ? "R8G8B8A8_UNORM"
                                                          : "other -- see vulkan_core.h");
    }
    p->color_format = pick.format;

    uint32_t count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && count > caps.maxImageCount) { count = caps.maxImageCount; }

    VkSwapchainKHR old = p->swapchain;
    VkSwapchainCreateInfoKHR ci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = p->surface,
        .minImageCount = count,
        .imageFormat = pick.format,
        .imageColorSpace = pick.colorSpace,
        .imageExtent = p->extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,
        .clipped = VK_TRUE,
        .oldSwapchain = old,
    };
    VK_CHECK(vkCreateSwapchainKHR(p->device, &ci, NULL, &p->swapchain));
    if (old) { vkDestroySwapchainKHR(p->device, old, NULL); }

    p->image_count = VK_PRESENT_MAX_IMAGES;
    VK_CHECK(vkGetSwapchainImagesKHR(p->device, p->swapchain, &p->image_count, p->images));

    for (uint32_t i = 0; i < p->image_count; i++) {
        VkImageViewCreateInfo vci = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = p->images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = p->color_format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        VK_CHECK(vkCreateImageView(p->device, &vci, NULL, &p->views[i]));

        VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(p->device, &sci, NULL, &p->render_done[i]));
    }

    if (p->flags & VK_PRESENT_DEPTH) { create_depth(p); }
}

static void destroy_swapchain_views(struct vk_presenter* p) {
    destroy_depth(p);
    for (uint32_t i = 0; i < p->image_count; i++) {
        vkDestroyImageView(p->device, p->views[i], NULL);
        vkDestroySemaphore(p->device, p->render_done[i], NULL);
    }
}

static void recreate_swapchain(struct vk_presenter* p) {
    vkDeviceWaitIdle(p->device);
    destroy_swapchain_views(p);
    create_swapchain(p);
    if (p->scene->resize) { p->scene->resize(p, p->extent, p->user); }
    p->resized = 0;
}

static void create_sync_and_commands(struct vk_presenter* p) {
    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = p->queue_family,
    };
    VK_CHECK(vkCreateCommandPool(p->device, &pci, NULL, &p->cmd_pool));

    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = p->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = VK_PRESENT_LANES,
    };
    VK_CHECK(vkAllocateCommandBuffers(p->device, &cai, p->cmd));

    for (int i = 0; i < VK_PRESENT_LANES; i++) {
        VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(p->device, &sci, NULL, &p->image_avail[i]));
        VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                 .flags = VK_FENCE_CREATE_SIGNALED_BIT};
        VK_CHECK(vkCreateFence(p->device, &fci, NULL, &p->in_flight[i]));
    }
}

/* ------------------------------------------------------------------ */
/* One frame (vulkan.c's draw_frame with the scene in the middle)     */
/* ------------------------------------------------------------------ */

static void draw_frame(struct vk_presenter* p) {
    uint32_t f = p->lane;

    VK_CHECK(vkWaitForFences(p->device, 1, &p->in_flight[f], VK_TRUE, UINT64_MAX));

    uint32_t img;
    VkResult r = vkAcquireNextImageKHR(p->device,
                                       p->swapchain,
                                       UINT64_MAX,
                                       p->image_avail[f],
                                       VK_NULL_HANDLE,
                                       &img);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || p->resized) {
        recreate_swapchain(p);
        return;
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) { die("acquire failed"); }

    VK_CHECK(vkResetFences(p->device, 1, &p->in_flight[f]));

    VkCommandBuffer cmd = p->cmd[f];
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    VkImageMemoryBarrier to_color = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = p->images[img],
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

    if (p->flags & VK_PRESENT_DEPTH) {
        /* The depth image is shared by both lanes, so the previous
         * frame may still be testing/writing it when this one starts:
         * order this frame's depth use after the last one's. UNDEFINED
         * as the old layout: we clear it, nothing to preserve. */
        VkImageMemoryBarrier to_depth = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = p->depth_image,
            .subresourceRange = {p->depth_aspect, 0, 1, 0, 1},
        };
        VkPipelineStageFlags depth_stages =
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        vkCmdPipelineBarrier(cmd, depth_stages, depth_stages, 0, 0, NULL, 0, NULL, 1, &to_depth);
    }

    struct vk_frame frame = {
        .cmd = cmd,
        .color_image = p->images[img],
        .color = p->views[img],
        .depth = p->depth_view,
        .image_index = img,
        .lane = f,
        .extent = p->extent,
    };
    p->scene->record(p, &frame, p->user);

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
        .pWaitSemaphores = &p->image_avail[f],
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &p->render_done[img],
    };
    VK_CHECK(vkQueueSubmit(p->queue, 1, &si, p->in_flight[f]));

    VkPresentInfoKHR pi = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &p->render_done[img],
        .swapchainCount = 1,
        .pSwapchains = &p->swapchain,
        .pImageIndices = &img,
    };
    r = vkQueuePresentKHR(p->queue, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || p->resized) {
        recreate_swapchain(p);
    } else if (r != VK_SUCCESS) {
        die("present failed");
    }

    p->lane = (f + 1) % VK_PRESENT_LANES;
}

/* ------------------------------------------------------------------ */

int vk_present_init(struct vk_presenter* p,
                    unsigned flags,
                    const struct vk_scene* scene,
                    void* user,
                    const char* title,
                    const char* app_id) {
    memset(p, 0, sizeof(*p));
    p->flags = flags;
    p->scene = scene;
    p->user = user;
    clock_gettime(CLOCK_MONOTONIC, &p->t0);

    if (app_init(&p->app, &present_backend, title, app_id) < 0) { return -1; }

    /* only after the configure/ack handshake may a buffer appear, and
     * the first present is that first buffer */
    while (!p->app.configured && wl_display_dispatch(p->app.display) != -1) {}

    create_instance(p);
    VkWaylandSurfaceCreateInfoKHR sci = {
        .sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
        .display = p->app.display,
        .surface = p->app.surface,
    };
    VK_CHECK(vkCreateWaylandSurfaceKHR(p->instance, &sci, NULL, &p->surface));
    pick_device(p);
    create_device(p);
    if (flags & VK_PRESENT_DEPTH) { pick_depth_format(p); }
    create_swapchain(p);
    create_sync_and_commands(p);

    p->scene->init(p, p->user);
    if (p->scene->resize) { p->scene->resize(p, p->extent, p->user); }
    return 0;
}

void vk_present_run(struct vk_presenter* p) {
    while (p->app.running) {
        if (wl_display_dispatch_pending(p->app.display) == -1) { break; }
        wl_display_flush(p->app.display);
        app_advance_clock(&p->app, now_ms(p));
        draw_frame(p);
    }
}

void vk_present_fini(struct vk_presenter* p) {
    vkDeviceWaitIdle(p->device);
    p->scene->fini(p, p->user);

    for (int i = 0; i < VK_PRESENT_LANES; i++) {
        vkDestroySemaphore(p->device, p->image_avail[i], NULL);
        vkDestroyFence(p->device, p->in_flight[i], NULL);
    }
    destroy_swapchain_views(p);
    vkDestroySwapchainKHR(p->device, p->swapchain, NULL);
    vkDestroyCommandPool(p->device, p->cmd_pool, NULL);
    vkDestroyDevice(p->device, NULL);
    vkDestroySurfaceKHR(p->instance, p->surface, NULL);
    vkDestroyInstance(p->instance, NULL);

    app_finish(&p->app);
}
