/* infer_ctx_vk.c -- the Vulkan domain, headless: instance + compute
 * queue with the external-memory / external-semaphore extensions
 * negotiated the way present/vk_present.c does it, two compute
 * pipelines, and exportable tensor memory.
 *
 * The export trick: Dawn imports dma-bufs as IMAGES only, so the
 * tensor's VkDeviceMemory is bound to a linear VkImage (R8G8B8A8_UNORM,
 * DRM_FORMAT_MOD_LINEAR, explicit row pitch) purely so that the fd
 * describes a texture -- and, when the driver permits aliasing, to a
 * VkBuffer as well, so an ordinary SSBO-writing compute shader fills
 * the tensor without knowing about the image. If aliasing is refused
 * (dedicated allocation required, disjoint memory types) the fallback
 * shader imageStores into the image instead; either way no host
 * memory is touched. Ownership crosses to the importer with a
 * VK_QUEUE_FAMILY_EXTERNAL barrier; completion is a SYNC_FD semaphore.
 */
#include "infer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <drm_fourcc.h>

#include "infer_util.h"

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

struct push {
    uint32_t img_w, img_h, pitch_floats, seed;
};

/* ------------------------------------------------------------------ */
/* bring-up                                                           */
/* ------------------------------------------------------------------ */

static int pick_device(struct infer_ctx_vk* c) {
    VkPhysicalDevice devs[8];
    uint32_t n = 8;
    VkResult r = vkEnumeratePhysicalDevices(c->instance, &n, devs);
    INFER_CHECK((r == VK_SUCCESS || r == VK_INCOMPLETE) && n > 0, "vk: no device");
    /* prefer a real GPU over llvmpipe */
    for (int pass = 0; pass < 2 && !c->phys; pass++) {
        for (uint32_t d = 0; d < n && !c->phys; d++) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devs[d], &props);
            if (pass == 0 && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) { continue; }
            VkQueueFamilyProperties fams[16];
            uint32_t fn = 16;
            vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &fn, fams);
            for (uint32_t f = 0; f < fn; f++) {
                if (fams[f].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    c->phys = devs[d];
                    c->queue_family = f;
                    fprintf(stderr, "vk: %s\n", props.deviceName);
                    break;
                }
            }
        }
    }
    INFER_CHECK(c->phys, "vk: no compute queue");
    vkGetPhysicalDeviceMemoryProperties(c->phys, &c->mem_props);
    return 0;
}

static int create_device(struct infer_ctx_vk* c) {
    static const char* const wanted[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    };
    int found[5] = {0};
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(c->phys, NULL, &n, NULL);
    VkExtensionProperties* props = malloc(n * sizeof *props);
    vkEnumerateDeviceExtensionProperties(c->phys, NULL, &n, props);
    const char* exts[8];
    uint32_t ext_count = 0;
    for (size_t w = 0; w < 5; w++) {
        for (uint32_t i = 0; i < n; i++) {
            if (strcmp(props[i].extensionName, wanted[w]) == 0) { found[w] = 1; }
        }
        if (found[w]) { exts[ext_count++] = wanted[w]; }
    }
    free(props);
    c->has_dmabuf_export = found[0] && found[1];
    c->has_drm_modifier = found[2];
    c->has_foreign_queue = found[3];
    c->has_sync_fd = found[4];
    fprintf(stderr,
            "vk: dma-buf export %s, drm modifier %s, sync fd %s\n",
            c->has_dmabuf_export ? "yes" : "NO",
            c->has_drm_modifier ? "yes" : "NO",
            c->has_sync_fd ? "yes" : "NO");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = c->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    /* whether this physical device can create a multiplanar image at all
     * -- the question infer_vk_frame_multiplanar_importable asks (on a
     * throwaway device: a failed import poisons the device it runs on).
     * Detected here, not enabled on the shared device, which never makes
     * one. */
    VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcr = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES};
    VkPhysicalDeviceFeatures2 have = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                      .pNext = &ycbcr};
    vkGetPhysicalDeviceFeatures2(c->phys, &have);
    c->has_ycbcr = ycbcr.samplerYcbcrConversion == VK_TRUE;
    VkDeviceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = ext_count,
        .ppEnabledExtensionNames = exts,
    };
    VK_CHECK(vkCreateDevice(c->phys, &ci, NULL, &c->device));
    vkGetDeviceQueue(c->device, c->queue_family, 0, &c->queue);
    c->get_memory_fd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(c->device, "vkGetMemoryFdKHR");
    c->get_semaphore_fd =
        (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(c->device, "vkGetSemaphoreFdKHR");
    c->import_semaphore_fd =
        (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(c->device, "vkImportSemaphoreFdKHR");
    c->get_fd_props =
        (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(c->device, "vkGetMemoryFdPropertiesKHR");
    if (!c->get_memory_fd) { c->has_dmabuf_export = 0; }
    if (!c->get_semaphore_fd || !c->import_semaphore_fd) { c->has_sync_fd = 0; }
    /* the import side: the same two memory extensions, the other direction */
    c->has_dmabuf_import = found[0] && found[1] && found[2] && c->get_fd_props != NULL;
    return 0;
}

static VkShaderModule load_shader(struct infer_ctx_vk* c, const char* path) {
    size_t size = 0;
    void* code = infer_read_file(path, &size);
    if (!code) { infer_die("vk: shader load failed"); }
    VkShaderModuleCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = code,
    };
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(c->device, &ci, NULL, &mod));
    free(code);
    return mod;
}

static VkPipeline make_pipeline(struct infer_ctx_vk* c, const char* path, VkPipelineLayout layout) {
    VkShaderModule mod = load_shader(c, path);
    VkComputePipelineCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = mod,
                  .pName = "main"},
        .layout = layout,
    };
    VkPipeline p;
    VK_CHECK(vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &p));
    vkDestroyShaderModule(c->device, mod, NULL);
    return p;
}

static int setup_common(struct infer_ctx_vk* c) {
    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = c->queue_family,
    };
    VK_CHECK(vkCreateCommandPool(c->device, &pci, NULL, &c->cmd_pool));
    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = c->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vkAllocateCommandBuffers(c->device, &cai, &c->cmd));

    /* two set layouts: the SSBO writer and the imageStore fallback */
    VkDescriptorSetLayoutBinding buf_b = {.binding = 0,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                          .descriptorCount = 1,
                                          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutBinding img_b = {.binding = 0,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          .descriptorCount = 1,
                                          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .bindingCount = 1,
                                           .pBindings = &buf_b};
    VK_CHECK(vkCreateDescriptorSetLayout(c->device, &lci, NULL, &c->buf_set_layout));
    lci.pBindings = &img_b;
    VK_CHECK(vkCreateDescriptorSetLayout(c->device, &lci, NULL, &c->img_set_layout));

    VkPushConstantRange pc = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(struct push)};
    VkPipelineLayoutCreateInfo pl = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .setLayoutCount = 1,
                                     .pSetLayouts = &c->buf_set_layout,
                                     .pushConstantRangeCount = 1,
                                     .pPushConstantRanges = &pc};
    VK_CHECK(vkCreatePipelineLayout(c->device, &pl, NULL, &c->buf_layout));
    pl.pSetLayouts = &c->img_set_layout;
    VK_CHECK(vkCreatePipelineLayout(c->device, &pl, NULL, &c->img_layout));
    c->gen_buf_pipeline = make_pipeline(c, SHADER_DIR "/infer_gen.comp.spv", c->buf_layout);
    c->gen_img_pipeline = make_pipeline(c, SHADER_DIR "/infer_gen_img.comp.spv", c->img_layout);

    VkDescriptorPoolSize sizes[2] = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 16},
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 16},
    };
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                      .maxSets = 32,
                                      .poolSizeCount = 2,
                                      .pPoolSizes = sizes};
    VK_CHECK(vkCreateDescriptorPool(c->device, &dpi, NULL, &c->pool));
    return 0;
}

int infer_ctx_vk_init(struct infer_ctx_vk* c) {
    memset(c, 0, sizeof *c);
    c->owned = 1;
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .pApplicationName = "hello-inference",
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app};
    VK_CHECK(vkCreateInstance(&ici, NULL, &c->instance));
    if (pick_device(c) < 0 || create_device(c) < 0) { return -1; }
    return setup_common(c);
}

int infer_ctx_vk_borrow(struct infer_ctx_vk* c,
                        VkInstance instance,
                        VkPhysicalDevice phys,
                        VkDevice device,
                        uint32_t queue_family,
                        VkQueue queue) {
    memset(c, 0, sizeof *c);
    c->instance = instance;
    c->phys = phys;
    c->device = device;
    c->queue_family = queue_family;
    c->queue = queue;
    vkGetPhysicalDeviceMemoryProperties(phys, &c->mem_props);
    /* the borrower must have enabled the extensions; probe the procs */
    c->get_memory_fd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    c->get_semaphore_fd = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
    c->import_semaphore_fd =
        (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkImportSemaphoreFdKHR");
    c->get_fd_props =
        (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdPropertiesKHR");
    c->has_dmabuf_export = c->get_memory_fd != NULL;
    c->has_drm_modifier = 1;
    c->has_sync_fd = c->get_semaphore_fd && c->import_semaphore_fd;
    c->has_dmabuf_import = c->get_fd_props != NULL;
    return setup_common(c);
}

void infer_ctx_vk_fini(struct infer_ctx_vk* c) {
    if (!c->device) { return; }
    vkDeviceWaitIdle(c->device);
    if (c->pool) { vkDestroyDescriptorPool(c->device, c->pool, NULL); }
    if (c->gen_buf_pipeline) { vkDestroyPipeline(c->device, c->gen_buf_pipeline, NULL); }
    if (c->gen_img_pipeline) { vkDestroyPipeline(c->device, c->gen_img_pipeline, NULL); }
    if (c->buf_layout) { vkDestroyPipelineLayout(c->device, c->buf_layout, NULL); }
    if (c->img_layout) { vkDestroyPipelineLayout(c->device, c->img_layout, NULL); }
    if (c->buf_set_layout) { vkDestroyDescriptorSetLayout(c->device, c->buf_set_layout, NULL); }
    if (c->img_set_layout) { vkDestroyDescriptorSetLayout(c->device, c->img_set_layout, NULL); }
    if (c->cmd_pool) { vkDestroyCommandPool(c->device, c->cmd_pool, NULL); }
    if (c->owned) {
        vkDestroyDevice(c->device, NULL);
        vkDestroyInstance(c->instance, NULL);
    }
    memset(c, 0, sizeof *c);
}

/* ------------------------------------------------------------------ */
/* the tensor: exportable memory as a linear image (+ buffer alias)    */
/* ------------------------------------------------------------------ */

static uint32_t find_memory_type(const struct infer_ctx_vk* c, uint32_t bits, VkMemoryPropertyFlags req) {
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (c->mem_props.memoryTypes[i].propertyFlags & req) == req) {
            return i;
        }
    }
    return UINT32_MAX;
}

/* per-tensor private state that the public struct does not carry */
struct vk_tensor_extra {
    VkDescriptorSet set;
    int use_image; /* writer: 0 = SSBO alias, 1 = imageStore */
};

static int alloc_tensor(struct infer_ctx_vk* c, const struct infer_desc* d, struct infer_tensor* t) {
    t->domain = INFER_DOMAIN_VK;
    t->desc = *d;
    t->owned = 1;
    t->ready.sync_fd = -1;
    t->released.sync_fd = -1;
    t->mem.vk.dmabuf_fd = -1;
    t->mem.vk.first_write = 1;
    INFER_CHECK(c->has_dmabuf_export && c->has_drm_modifier, "vk: dma-buf export unavailable");
    struct infer_mem_vk* m = &t->mem.vk;
    struct vk_tensor_extra* x = calloc(1, sizeof *x);
    t->priv = x;
    t->priv_free = free;

    /* 1. can the driver make a linear, exportable RGBA8 image, and with
     *    storage usage (for the fallback writer)? */
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    VkExternalImageFormatProperties ext_props = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
    VkImageFormatProperties2 fprops = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                                       .pNext = &ext_props};
    for (int attempt = 0; attempt < 2; attempt++) {
        VkPhysicalDeviceImageDrmFormatModifierInfoEXT mod_info = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
            .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VkPhysicalDeviceExternalImageFormatInfo ext_info = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
            .pNext = &mod_info,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        };
        VkPhysicalDeviceImageFormatInfo2 info = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
            .pNext = &ext_info,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .type = VK_IMAGE_TYPE_2D,
            .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
            .usage = usage,
        };
        VkResult r = vkGetPhysicalDeviceImageFormatProperties2(c->phys, &info, &fprops);
        if (r == VK_SUCCESS) { break; }
        INFER_CHECK(attempt == 0, "vk: linear exportable RGBA8 image unsupported (%d)", (int)r);
        usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT; /* retry without storage */
    }
    INFER_CHECK(ext_props.externalMemoryProperties.externalMemoryFeatures &
                    VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT,
                "vk: image memory not exportable as dma-buf");
    int storage_ok = (usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0;

    /* 2. the image, with an explicit linear layout; try pitches until
     *    the driver accepts one */
    const uint32_t pitches[] = {infer_align(d->img_w * 4, 64), d->img_w * 4, infer_align(d->img_w * 4, 256)};
    VkResult r = VK_ERROR_UNKNOWN;
    for (size_t p = 0; p < 3 && r != VK_SUCCESS; p++) {
        VkSubresourceLayout plane = {.offset = 0, .rowPitch = pitches[p]};
        VkImageDrmFormatModifierExplicitCreateInfoEXT mod = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
            .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
            .drmFormatModifierPlaneCount = 1,
            .pPlaneLayouts = &plane,
        };
        VkExternalMemoryImageCreateInfo ext = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            .pNext = &mod,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        };
        VkImageCreateInfo ici = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &ext,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {d->img_w, d->img_h, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        r = vkCreateImage(c->device, &ici, NULL, &m->image);
        if (r == VK_SUCCESS) {
            t->desc.row_pitch_bytes = pitches[p];
            t->desc.drm_modifier = DRM_FORMAT_MOD_LINEAR;
            t->desc.planes = 1;
            t->desc.plane_offset[0] = 0;
            t->desc.plane_pitch[0] = pitches[p];
        }
    }
    INFER_CHECK(r == VK_SUCCESS, "vk: vkCreateImage(linear, exportable) failed: %d", (int)r);

    VkMemoryDedicatedRequirements ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 img_req = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &ded};
    VkImageMemoryRequirementsInfo2 iri = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                                          .image = m->image};
    vkGetImageMemoryRequirements2(c->device, &iri, &img_req);

    /* 3. the buffer alias */
    VkDeviceSize bytes = (VkDeviceSize)t->desc.row_pitch_bytes * d->img_h;
    VkExternalMemoryBufferCreateInfo bext = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &bext,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer alias = VK_NULL_HANDLE;
    VkMemoryRequirements buf_req = {0};
    if (vkCreateBuffer(c->device, &bci, NULL, &alias) == VK_SUCCESS) {
        vkGetBufferMemoryRequirements(c->device, alias, &buf_req);
    }
    uint32_t bits = img_req.memoryRequirements.memoryTypeBits;
    int alias_ok = alias != VK_NULL_HANDLE && !ded.requiresDedicatedAllocation &&
                   (bits & buf_req.memoryTypeBits) != 0 &&
                   buf_req.size <= img_req.memoryRequirements.size + buf_req.alignment;
    if (alias_ok) { bits &= buf_req.memoryTypeBits; }
    if (!alias_ok && !storage_ok) {
        if (alias) { vkDestroyBuffer(c->device, alias, NULL); }
        INFER_CHECK(0, "vk: neither buffer aliasing nor storage image available");
    }
    if (!alias_ok && alias) {
        vkDestroyBuffer(c->device, alias, NULL);
        alias = VK_NULL_HANDLE;
    }

    /* 4. exportable memory, host-visible if the driver lets us */
    VkDeviceSize size = img_req.memoryRequirements.size;
    if (alias_ok && buf_req.size > size) { size = buf_req.size; }
    uint32_t type = find_memory_type(c, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) { type = find_memory_type(c, bits, 0); }
    INFER_CHECK(type != UINT32_MAX, "vk: no memory type");
    m->mem_type = type;
    m->host_visible =
        (c->mem_props.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    VkMemoryDedicatedAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                         .image = m->image};
    VkExportMemoryAllocateInfo exp = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .pNext = (!alias_ok && ded.requiresDedicatedAllocation) ? &dai : NULL,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .pNext = &exp,
                                .allocationSize = size,
                                .memoryTypeIndex = type};
    VK_CHECK(vkAllocateMemory(c->device, &mai, NULL, &m->memory));
    VK_CHECK(vkBindImageMemory(c->device, m->image, m->memory, 0));
    if (alias_ok) { VK_CHECK(vkBindBufferMemory(c->device, alias, m->memory, 0)); }
    m->buffer = alias;
    m->alias_ok = alias_ok;
    m->size = size;
    x->use_image = !alias_ok;

    VkMemoryGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                                .memory = m->memory,
                                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VK_CHECK(c->get_memory_fd(c->device, &gfi, &m->dmabuf_fd));

    /* 5. descriptor set for whichever writer applies */
    VkDescriptorSetAllocateInfo dsa = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = c->pool,
        .descriptorSetCount = 1,
        .pSetLayouts = alias_ok ? &c->buf_set_layout : &c->img_set_layout,
    };
    VK_CHECK(vkAllocateDescriptorSets(c->device, &dsa, &x->set));
    if (alias_ok) {
        VkDescriptorBufferInfo bi = {.buffer = m->buffer, .offset = 0, .range = bytes};
        VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                  .dstSet = x->set,
                                  .dstBinding = 0,
                                  .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                  .pBufferInfo = &bi};
        vkUpdateDescriptorSets(c->device, 1, &w, 0, NULL);
    } else {
        VkImageViewCreateInfo vci = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = m->image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        VK_CHECK(vkCreateImageView(c->device, &vci, NULL, &m->storage_view));
        VkDescriptorImageInfo ii = {.imageView = m->storage_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                  .dstSet = x->set,
                                  .dstBinding = 0,
                                  .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                  .pImageInfo = &ii};
        vkUpdateDescriptorSets(c->device, 1, &w, 0, NULL);
    }

    /* 6. sync objects */
    VkExportSemaphoreCreateInfo esi = {.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
                                       .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
    VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                 .pNext = c->has_sync_fd ? &esi : NULL};
    VK_CHECK(vkCreateSemaphore(c->device, &sci, NULL, &m->done));
    sci.pNext = NULL;
    VK_CHECK(vkCreateSemaphore(c->device, &sci, NULL, &m->release));
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(c->device, &fci, NULL, &m->fence));

    if (!infer_verbose) { return 0; }
    fprintf(stderr,
            "vk: tensor %ux%u pitch %u, writer = %s, memory %s"
            " (type %u flags 0x%x, heap %u flags 0x%x, %.0f MiB)\n",
            d->img_w,
            d->img_h,
            t->desc.row_pitch_bytes,
            alias_ok ? "SSBO alias" : "imageStore",
            m->host_visible ? "host-visible" : "device-local",
            m->mem_type,
            (unsigned)c->mem_props.memoryTypes[m->mem_type].propertyFlags,
            c->mem_props.memoryTypes[m->mem_type].heapIndex,
            (unsigned)c->mem_props.memoryHeaps[c->mem_props.memoryTypes[m->mem_type].heapIndex].flags,
            (double)c->mem_props.memoryHeaps[c->mem_props.memoryTypes[m->mem_type].heapIndex].size
                / (1024.0 * 1024.0));
    return 0;
}

int infer_vk_alloc(struct infer_ctx_vk* c, const struct infer_desc* d, struct infer_tensor* t) {
    return t->mem.vk.memory ? 0 : alloc_tensor(c, d, t);
}

int infer_vk_gen(struct infer_ctx_vk* c,
                 const struct infer_desc* d,
                 uint32_t seed,
                 struct infer_tensor* t) {
    if (infer_vk_alloc(c, d, t) < 0) { return -1; }
    struct infer_mem_vk* m = &t->mem.vk;
    struct vk_tensor_extra* x = t->priv;

    /* the consumer's fence -> a semaphore this submit waits on */
    VkSemaphore wait_sem = VK_NULL_HANDLE;
    if (t->released.sync_fd >= 0) {
        if (c->has_sync_fd) {
            VkImportSemaphoreFdInfoKHR isi = {
                .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
                .semaphore = m->release,
                .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
                .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
                .fd = t->released.sync_fd,
            };
            if (c->import_semaphore_fd(c->device, &isi) == VK_SUCCESS) {
                wait_sem = m->release;
                t->released.sync_fd = -1; /* Vulkan owns the fd now */
            }
        }
        if (wait_sem == VK_NULL_HANDLE) {
            infer_wait_sync_fd(t->released.sync_fd, 1000);
            infer_sync_reset(&t->released);
        }
    }
    if (!m->first_write) {
        VK_CHECK(vkWaitForFences(c->device, 1, &m->fence, VK_TRUE, UINT64_MAX));
    }
    VK_CHECK(vkResetFences(c->device, 1, &m->fence));

    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(c->cmd, &bi));

    /* acquire from the external reader (or the first-ever layout) */
    VkImageMemoryBarrier ib = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = m->first_write ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = m->first_write ? VK_QUEUE_FAMILY_IGNORED : VK_QUEUE_FAMILY_EXTERNAL,
        .dstQueueFamilyIndex = m->first_write ? VK_QUEUE_FAMILY_IGNORED : c->queue_family,
        .image = m->image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkBufferMemoryBarrier bb = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = m->first_write ? VK_QUEUE_FAMILY_IGNORED : VK_QUEUE_FAMILY_EXTERNAL,
        .dstQueueFamilyIndex = m->first_write ? VK_QUEUE_FAMILY_IGNORED : c->queue_family,
        .buffer = m->buffer,
        .size = VK_WHOLE_SIZE,
    };
    vkCmdPipelineBarrier(c->cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL,
                         x->use_image ? 0 : 1, &bb,
                         x->use_image || m->first_write ? 1 : 0, &ib);

    struct push pc = {t->desc.img_w, t->desc.img_h, t->desc.row_pitch_bytes / 4, seed};
    VkPipelineLayout layout = x->use_image ? c->img_layout : c->buf_layout;
    vkCmdBindPipeline(c->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      x->use_image ? c->gen_img_pipeline : c->gen_buf_pipeline);
    vkCmdBindDescriptorSets(c->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &x->set, 0, NULL);
    vkCmdPushConstants(c->cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(c->cmd, (t->desc.img_w + 15) / 16, (t->desc.img_h + 15) / 16, 1);

    /* release to the external reader */
    ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ib.dstAccessMask = 0;
    ib.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = c->queue_family;
    ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bb.dstAccessMask = 0;
    bb.srcQueueFamilyIndex = c->queue_family;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    vkCmdPipelineBarrier(c->cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, NULL,
                         x->use_image ? 0 : 1, &bb,
                         1, &ib);
    VK_CHECK(vkEndCommandBuffer(c->cmd));

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = wait_sem ? 1 : 0,
        .pWaitSemaphores = &wait_sem,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &c->cmd,
        .signalSemaphoreCount = c->has_sync_fd ? 1 : 0,
        .pSignalSemaphores = &m->done,
    };
    VK_CHECK(vkQueueSubmit(c->queue, 1, &si, m->fence));
    m->first_write = 0;

    if (c->has_sync_fd) {
        VkSemaphoreGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                                       .semaphore = m->done,
                                       .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
        int fd = -1;
        if (c->get_semaphore_fd(c->device, &gfi, &fd) != VK_SUCCESS) { fd = -1; }
        infer_sync_set(&t->ready, fd);
    } else {
        VK_CHECK(vkWaitForFences(c->device, 1, &m->fence, VK_TRUE, UINT64_MAX));
        infer_sync_set(&t->ready, -1);
    }
    return 0;
}

static int host_coherent(const struct infer_ctx_vk* c, const struct infer_mem_vk* m) {
    return (c->mem_props.memoryTypes[m->mem_type].propertyFlags &
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
}

/* image rows (pitch apart) <-> packed floats; the image's last row may
 * hold more texels than there are elements left */
static void rows_out(float* dst, const void* img, uint32_t pitch, const struct infer_desc* d) {
    size_t n = infer_desc_elements(d);
    for (size_t y = 0; y * d->img_w < n; y++) {
        size_t cnt = n - y * d->img_w < d->img_w ? n - y * d->img_w : d->img_w;
        memcpy(dst + y * d->img_w, (const char*)img + y * pitch, cnt * 4);
    }
}
static void rows_in(void* img, const float* src, uint32_t pitch, const struct infer_desc* d) {
    size_t n = infer_desc_elements(d);
    for (size_t y = 0; y * d->img_w < n; y++) {
        size_t cnt = n - y * d->img_w < d->img_w ? n - y * d->img_w : d->img_w;
        memcpy((char*)img + y * pitch, src + y * d->img_w, cnt * 4);
    }
}

int infer_vk_readback(struct infer_ctx_vk* c, const struct infer_tensor* t, float* dst) {
    const struct infer_mem_vk* m = &t->mem.vk;
    /* whoever wrote it last: a foreign writer (Dawn) whose fence is the
     * sync file on `ready`, or our own generator (its fence; never
     * submitted if nobody generated into this tensor) */
    if (t->ready.sync_fd >= 0) {
        INFER_CHECK(infer_wait_sync_fd(t->ready.sync_fd, 2000) == 0, "vk: ready fence timed out");
    }
    if (!m->first_write) { VK_CHECK(vkWaitForFences(c->device, 1, &m->fence, VK_TRUE, UINT64_MAX)); }
    uint32_t pitch = t->desc.row_pitch_bytes;
    if (m->host_visible) {
        void* p = NULL;
        VK_CHECK(vkMapMemory(c->device, m->memory, 0, VK_WHOLE_SIZE, 0, &p));
        if (!host_coherent(c, m)) {
            VkMappedMemoryRange r = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                     .memory = m->memory,
                                     .size = VK_WHOLE_SIZE};
            VK_CHECK(vkInvalidateMappedMemoryRanges(c->device, 1, &r));
        }
        rows_out(dst, p, pitch, &t->desc);
        vkUnmapMemory(c->device, m->memory);
        return 0;
    }
    /* device-local: copy into a host-visible staging buffer */
    VkDeviceSize bytes = (VkDeviceSize)pitch * t->desc.img_h;
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = bytes,
                              .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VkBuffer staging;
    VK_CHECK(vkCreateBuffer(c->device, &bci, NULL, &staging));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, staging, &req);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = find_memory_type(
            c, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };
    VkDeviceMemory smem;
    VK_CHECK(vkAllocateMemory(c->device, &mai, NULL, &smem));
    VK_CHECK(vkBindBufferMemory(c->device, staging, smem, 0));

    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(c->cmd, &bi));
    if (m->alias_ok) {
        VkBufferCopy region = {.size = bytes};
        vkCmdCopyBuffer(c->cmd, m->buffer, staging, 1, &region);
    } else {
        VkBufferImageCopy region = {.bufferRowLength = pitch / 4,
                                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                    .imageExtent = {t->desc.img_w, t->desc.img_h, 1}};
        vkCmdCopyImageToBuffer(c->cmd, m->image, VK_IMAGE_LAYOUT_GENERAL, staging, 1, &region);
    }
    VK_CHECK(vkEndCommandBuffer(c->cmd));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &c->cmd};
    VK_CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c->queue));
    void* p = NULL;
    VK_CHECK(vkMapMemory(c->device, smem, 0, VK_WHOLE_SIZE, 0, &p));
    rows_out(dst, p, pitch, &t->desc);
    vkUnmapMemory(c->device, smem);
    vkDestroyBuffer(c->device, staging, NULL);
    vkFreeMemory(c->device, smem, NULL);
    return 0;
}

/* Host floats into the exportable memory (an output a Vulkan consumer
 * owns, produced by the CPU engine): a map when the memory is host-
 * visible, a staging copy when it is device-local. */
int infer_vk_upload(struct infer_ctx_vk* c, const float* src, struct infer_tensor* t) {
    struct infer_mem_vk* m = &t->mem.vk;
    uint32_t pitch = t->desc.row_pitch_bytes;
    uint32_t w = t->desc.img_w, h = t->desc.img_h;
    /* our own writer may still be running, and a foreign reader may
     * still be reading (honoured on the host here; infer_vk_gen does it
     * on the GPU timeline) */
    if (!m->first_write) { VK_CHECK(vkWaitForFences(c->device, 1, &m->fence, VK_TRUE, UINT64_MAX)); }
    if (t->released.sync_fd >= 0) {
        infer_wait_sync_fd(t->released.sync_fd, 1000);
        infer_sync_reset(&t->released);
    }
    if (m->host_visible) {
        void* p = NULL;
        VK_CHECK(vkMapMemory(c->device, m->memory, 0, VK_WHOLE_SIZE, 0, &p));
        rows_in(p, src, pitch, &t->desc);
        if (!host_coherent(c, m)) {
            VkMappedMemoryRange r = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                     .memory = m->memory,
                                     .size = VK_WHOLE_SIZE};
            VK_CHECK(vkFlushMappedMemoryRanges(c->device, 1, &r));
        }
        vkUnmapMemory(c->device, m->memory);
        infer_sync_set(&t->ready, -1); /* host writes are complete on return */
        return 0;
    }
    /* device-local: stage through host-visible memory */
    VkDeviceSize bytes = (VkDeviceSize)pitch * h;
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = bytes,
                              .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    VkBuffer staging;
    VK_CHECK(vkCreateBuffer(c->device, &bci, NULL, &staging));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, staging, &req);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = find_memory_type(
            c, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };
    VkDeviceMemory smem;
    VK_CHECK(vkAllocateMemory(c->device, &mai, NULL, &smem));
    VK_CHECK(vkBindBufferMemory(c->device, staging, smem, 0));
    void* p = NULL;
    VK_CHECK(vkMapMemory(c->device, smem, 0, VK_WHOLE_SIZE, 0, &p));
    rows_in(p, src, pitch, &t->desc);
    vkUnmapMemory(c->device, smem);

    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(c->cmd, &bi));
    if (m->alias_ok) {
        VkBufferCopy region = {.size = bytes};
        vkCmdCopyBuffer(c->cmd, staging, m->buffer, 1, &region);
    } else {
        /* the image path: to GENERAL first (UNDEFINED discards, and we
         * overwrite the whole image anyway) */
        VkImageMemoryBarrier ib = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = m->first_write ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m->image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        vkCmdPipelineBarrier(c->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, NULL, 0, NULL, 1, &ib);
        VkBufferImageCopy region = {.bufferRowLength = pitch / 4,
                                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                    .imageExtent = {w, h, 1}};
        vkCmdCopyBufferToImage(c->cmd, staging, m->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    }
    VK_CHECK(vkEndCommandBuffer(c->cmd));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &c->cmd};
    VK_CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c->queue));
    vkDestroyBuffer(c->device, staging, NULL);
    vkFreeMemory(c->device, smem, NULL);
    infer_sync_set(&t->ready, -1);
    return 0;
}

void infer_vk_release(struct infer_ctx_vk* c, struct infer_tensor* t) {
    if (!t->owned) { return; }
    struct infer_mem_vk* m = &t->mem.vk;
    if (!m->memory) { return; }
    vkQueueWaitIdle(c->queue);
    struct vk_tensor_extra* x = t->priv;
    if (x && x->set) { vkFreeDescriptorSets(c->device, c->pool, 1, &x->set); }
    if (m->fence) { vkDestroyFence(c->device, m->fence, NULL); }
    if (m->done) { vkDestroySemaphore(c->device, m->done, NULL); }
    if (m->release) { vkDestroySemaphore(c->device, m->release, NULL); }
    if (m->storage_view) { vkDestroyImageView(c->device, m->storage_view, NULL); }
    if (m->buffer) { vkDestroyBuffer(c->device, m->buffer, NULL); }
    if (m->image) { vkDestroyImage(c->device, m->image, NULL); }
    if (m->dmabuf_fd >= 0) { close(m->dmabuf_fd); }
    vkFreeMemory(c->device, m->memory, NULL);
}

/* ------------------------------------------------------------------ */
/* the camera frame, imported per plane (see infer.h)                  */
/* ------------------------------------------------------------------ */

struct infer_vk_frame_import {
    VkDevice dev; /* for the cache-style destroy, which gets no ctx */
    VkImage image[2];
    VkDeviceMemory mem[2];
    VkImageView view[2];
    int nplanes;
    int first; /* first acquire transitions from UNDEFINED */
};

/* one plane of the frame as its own single-plane LINEAR image over the
 * plane's fd at the plane's offset -- the multiplanar format (and its
 * SamplerYcbcrConversion baggage) never enters the picture, because the
 * shader does the YUV maths itself */
static int import_plane(struct infer_ctx_vk* c,
                        struct infer_vk_frame_import* im,
                        int i,
                        VkFormat fmt,
                        uint32_t w,
                        uint32_t h,
                        int fd,
                        uint32_t offset,
                        uint32_t pitch) {
    VkSubresourceLayout plane = {.offset = offset, .rowPitch = pitch};
    VkImageDrmFormatModifierExplicitCreateInfoEXT mod = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
        .drmFormatModifierPlaneCount = 1,
        .pPlaneLayouts = &plane,
    };
    VkExternalMemoryImageCreateInfo ext = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = &mod,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &ext,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = fmt,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (vkCreateImage(c->device, &ici, NULL, &im->image[i]) != VK_SUCCESS) {
        fprintf(stderr, "vk: frame plane %d image (fmt %d, %ux%u) refused\n", i, (int)fmt, w, h);
        return -1;
    }
    VkMemoryFdPropertiesKHR fdp = {.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    if (c->get_fd_props(c->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fdp) !=
        VK_SUCCESS) {
        fprintf(stderr, "vk: frame fd not importable\n");
        return -1;
    }
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c->device, im->image[i], &req);
    uint32_t bits = req.memoryTypeBits & fdp.memoryTypeBits;
    if (!bits) {
        fprintf(stderr, "vk: no memory type can import the frame's dma-buf\n");
        return -1;
    }
    /* an import's allocationSize is the DMA-BUF'S size, not the image's
     * requirement (VUID 01742): the fd backs the whole buffer, and this
     * plane's image is a window into it at `offset` */
    off_t fd_size = lseek(fd, 0, SEEK_END);
    VkDeviceSize alloc_size = fd_size > 0 ? (VkDeviceSize)fd_size : req.size;
    if (alloc_size < req.size) { alloc_size = req.size; }
    /* dedicated, as the presenter's import and the NVIDIA war story in
     * infer/patches/dawn both counsel; the fd is dup'd because a
     * successful import consumes it */
    VkMemoryDedicatedAllocateInfo dedicated = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .image = im->image[i],
    };
    VkImportMemoryFdInfoKHR import = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .pNext = &dedicated,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = dup(fd),
    };
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import,
        .allocationSize = alloc_size,
        .memoryTypeIndex = find_memory_type(c, bits, 0),
    };
    if (infer_verbose) {
        fprintf(stderr, "vk: frame plane %d: fd size %lld, image req %llu, type %u\n", i,
                (long long)fd_size, (unsigned long long)req.size, mai.memoryTypeIndex);
    }
    if (vkAllocateMemory(c->device, &mai, NULL, &im->mem[i]) != VK_SUCCESS) {
        fprintf(stderr, "vk: frame dma-buf import failed (plane %d)\n", i);
        close(import.fd);
        return -1;
    }
    if (vkBindImageMemory(c->device, im->image[i], im->mem[i], 0) != VK_SUCCESS) { return -1; }
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = im->image[i],
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = fmt,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    if (vkCreateImageView(c->device, &vci, NULL, &im->view[i]) != VK_SUCCESS) { return -1; }
    return 0;
}

struct infer_vk_frame_import* infer_vk_frame_import_create(struct infer_ctx_vk* c,
                                                           const struct infer_frame* f) {
    if (!c->device || !c->has_dmabuf_import || !c->has_drm_modifier) {
        fprintf(stderr, "vk: no dma-buf import on this device\n");
        return NULL;
    }
    if (f->drm_modifier != DRM_FORMAT_MOD_LINEAR) {
        fprintf(stderr, "vk: only LINEAR frames are imported per plane\n");
        return NULL;
    }
    struct infer_vk_frame_import* im = calloc(1, sizeof *im);
    if (!im) { return NULL; }
    im->dev = c->device;
    im->first = 1;
    int r = -1;
    if (f->drm_format == DRM_FORMAT_NV12 && f->planes == 2) {
        im->nplanes = 2;
        const int fd1 = f->dmabuf_fd[1] >= 0 ? f->dmabuf_fd[1] : f->dmabuf_fd[0];
        r = import_plane(c, im, 0, VK_FORMAT_R8_UNORM, f->width, f->height, f->dmabuf_fd[0],
                         f->offset[0], f->pitch[0]);
        if (r == 0) {
            r = import_plane(c, im, 1, VK_FORMAT_R8G8_UNORM, f->width / 2, f->height / 2, fd1,
                             f->offset[1], f->pitch[1]);
        }
    } else if (f->drm_format == DRM_FORMAT_ABGR8888 && f->planes == 1) {
        /* the YUYV-as-ABGR substitution: hand_frame_import_desc's output */
        im->nplanes = 1;
        r = import_plane(c, im, 0, VK_FORMAT_R8G8B8A8_UNORM, f->width, f->height, f->dmabuf_fd[0],
                         f->offset[0], f->pitch[0]);
    } else {
        fprintf(stderr, "vk: frame fourcc 0x%08x not importable here\n", f->drm_format);
    }
    if (r < 0) {
        infer_vk_frame_import_destroy(im);
        return NULL;
    }
    if (infer_verbose) {
        fprintf(stderr, "vk: frame imported, %d plane image(s) %ux%u\n", im->nplanes, f->width,
                f->height);
    }
    return im;
}

VkImageView infer_vk_frame_plane(const struct infer_vk_frame_import* im, int plane) {
    return im && plane >= 0 && plane < im->nplanes ? im->view[plane] : VK_NULL_HANDLE;
}

int infer_vk_frame_acquire(struct infer_ctx_vk* c,
                           struct infer_vk_frame_import* im,
                           VkCommandBuffer cmd) {
    /* A fresh layout barrier every pass: the frame's bytes change outside
     * Vulkan between uses. First use establishes GENERAL from UNDEFINED
     * (linear imported memory; nothing is re-tiled and nothing was
     * discarded, measured); no queue-family ownership transfer -- the
     * writer is a camera or a CPU, which never executes a Vulkan release,
     * so an acquire would pair with nothing. This is also what Dawn's
     * dma-buf access does. Visibility of the outside writes comes from
     * the submission itself plus, for CPU writers on x86, the producer's
     * clflush (README §20's dma-buf story). */
    (void)c;
    VkImageMemoryBarrier b[2];
    for (int i = 0; i < im->nplanes; i++) {
        b[i] = (VkImageMemoryBarrier){
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = im->first ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = im->image[i],
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL,
                         (uint32_t)im->nplanes, b);
    im->first = 0;
    return 0;
}

void infer_vk_frame_import_destroy(void* p) {
    struct infer_vk_frame_import* im = p;
    if (!im) { return; }
    /* the caller has waited the pass's fence (hand_frame_vk_wait /
     * pass destroy); views before images before memory, as everywhere */
    for (int i = 0; i < im->nplanes; i++) {
        if (im->view[i]) { vkDestroyImageView(im->dev, im->view[i], NULL); }
        if (im->image[i]) { vkDestroyImage(im->dev, im->image[i], NULL); }
        if (im->mem[i]) { vkFreeMemory(im->dev, im->mem[i], NULL); }
    }
    free(im);
}

/* The import Dawn makes of an NV12 dma-buf (SharedTextureMemoryVk.cpp,
 * Create(DmaBufDescriptor)), replayed step by step: one
 * G8_B8R8_2PLANE_420 image with the frame's explicit plane layout, the
 * MUTABLE flag and R8/RG8 view-format list a sampled multiplanar image
 * gets, the fd's memory types intersected with the image's, a dedicated
 * allocation of the image's requirement (Dawn's choice of size, patch
 * 0001's choice of dedicated). Every type in the intersection is tried,
 * so "refused" means no type at all -- the one answer safe to act on.
 *
 * Run on a THROWAWAY VkDevice, and this is why: a FAILED dma-buf import
 * poisons the device it ran on. Measured on the 2080 Ti / 610.43 with a
 * raw-Vulkan replay -- after the refused import, every later
 * vkAllocateMemory on that VkDevice returns OUT_OF_DEVICE_MEMORY,
 * exportable or plain, and vkDeviceWaitIdle does not clear it; a device
 * created afterward is unaffected. Asking the question on our shared
 * device would take down the very device the vk feed then writes on, so
 * the question is put to a device made for it and destroyed after -- the
 * poison dies with it, and the verdict (a property of the driver and the
 * frame layout, not the device instance) is the same. */
int infer_vk_frame_multiplanar_importable(struct infer_ctx_vk* c, const struct infer_frame* f) {
    if (!c->instance || !c->phys || !c->has_dmabuf_import || !c->has_drm_modifier || !c->has_ycbcr) {
        return -1;
    }
    if (f->drm_format != DRM_FORMAT_NV12 || f->planes != 2 || f->dmabuf_fd[0] < 0) { return -1; }
    const uint32_t key[6] = {f->width,    f->height,   f->offset[0],
                             f->offset[1], f->pitch[0], f->pitch[1]};
    if (c->mp_probe_result != 0 && memcmp(key, c->mp_probe_key, sizeof key) == 0) {
        return c->mp_probe_result > 0 ? 1 : 0;
    }

    /* the throwaway device, on the same physical device: the memory and
     * modifier extensions the import needs, and the ycbcr feature a
     * multiplanar image needs at device creation */
    const char* exts[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
    };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueFamilyIndex = c->queue_family,
                                   .queueCount = 1,
                                   .pQueuePriorities = &prio};
    VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcr = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
        .samplerYcbcrConversion = VK_TRUE};
    VkPhysicalDeviceFeatures2 feat = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                      .pNext = &ycbcr};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .pNext = &feat,
                              .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = 3,
                              .ppEnabledExtensionNames = exts};
    VkDevice pdev = VK_NULL_HANDLE;
    if (vkCreateDevice(c->phys, &dci, NULL, &pdev) != VK_SUCCESS) { return -1; }
    PFN_vkGetMemoryFdPropertiesKHR get_fdp =
        (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(pdev, "vkGetMemoryFdPropertiesKHR");

    int ok = 0;
    const char* why = "vkCreateImage";
    VkFormat views[2] = {VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM};
    VkImageFormatListCreateInfo flist = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
                                         .viewFormatCount = 2,
                                         .pViewFormats = views};
    VkSubresourceLayout planes[2] = {
        {.offset = f->offset[0], .rowPitch = f->pitch[0]},
        {.offset = f->offset[1], .rowPitch = f->pitch[1]},
    };
    VkImageDrmFormatModifierExplicitCreateInfoEXT mod = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .pNext = &flist,
        .drmFormatModifier = f->drm_modifier,
        .drmFormatModifierPlaneCount = 2,
        .pPlaneLayouts = planes,
    };
    VkExternalMemoryImageCreateInfo ext = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = &mod,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &ext,
        .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
        .extent = {f->width, f->height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage img = VK_NULL_HANDLE;
    if (get_fdp && vkCreateImage(pdev, &ici, NULL, &img) == VK_SUCCESS) {
        VkMemoryFdPropertiesKHR fdp = {.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(pdev, img, &req);
        uint32_t bits = 0;
        if (get_fdp(pdev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, f->dmabuf_fd[0], &fdp) ==
            VK_SUCCESS) {
            bits = req.memoryTypeBits & fdp.memoryTypeBits;
        }
        why = bits ? "vkAllocateMemory" : "no memory type";
        for (uint32_t t = 0; t < c->mem_props.memoryTypeCount; t++) {
            if (!(bits & (1u << t))) { continue; }
            VkMemoryDedicatedAllocateInfo dedicated = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                .image = img,
            };
            VkImportMemoryFdInfoKHR import = {
                .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
                .pNext = &dedicated,
                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                .fd = dup(f->dmabuf_fd[0]),
            };
            VkMemoryAllocateInfo mai = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = &import,
                .allocationSize = req.size,
                .memoryTypeIndex = t,
            };
            struct stat before, after;
            const int known = fstat(import.fd, &before) == 0;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            if (vkAllocateMemory(pdev, &mai, NULL, &mem) == VK_SUCCESS) {
                /* the import consumed the fd; a bound image is spent either
                 * way, so this is the last attempt */
                ok = vkBindImageMemory(pdev, img, mem, 0) == VK_SUCCESS;
                if (!ok) { why = "vkBindImageMemory"; }
                vkFreeMemory(pdev, mem, NULL);
                break;
            }
            /* A failed import leaves the fd with us (the spec transfers it
             * on success only) -- except that this driver closes it
             * anyway (NVIDIA 610.43, on this dedicated import) and may
             * have opened something of its own under the number since:
             * closing THAT would hit an unrelated fd. A dma-buf's inode is
             * its identity; close only a descriptor still ours. (The
             * device is thrown away next, so the poison the failure leaves
             * behind is discarded with it.) */
            if (known && fstat(import.fd, &after) == 0 && after.st_ino == before.st_ino &&
                after.st_dev == before.st_dev) {
                close(import.fd);
            }
        }
        vkDestroyImage(pdev, img, NULL);
    }
    vkDestroyDevice(pdev, NULL);
    memcpy(c->mp_probe_key, key, sizeof key);
    c->mp_probe_result = ok ? 1 : -1;
    if (infer_verbose || !ok) {
        fprintf(stderr, "vk: the frame as one multiplanar linear image: %s%s%s\n",
                ok ? "imports" : "REFUSED at ", ok ? "" : why,
                ok ? "" : " -- Dawn's frame import would not survive this");
    }
    return ok;
}

/* ------------------------------------------------------------------ */
/* what a Vulkan importer will accept                                  */
/* ------------------------------------------------------------------ */

/* Dawn's Linux backend is Vulkan, so the modifiers this driver lists
 * for R8G8B8A8_UNORM are exactly the ones a WebGPU import can accept.
 * The GL domain needs this to allocate a tensor an importer will take:
 * left to itself, NVIDIA's GBM hands back a block-linear layout that is
 * renderable but absent from this list. Standalone instance, because a
 * gl -> webgpu run never brings up the Vulkan domain. */
int infer_vk_importable_modifiers(uint64_t* out, int max) {
    /* Queried once and kept: on NVIDIA, destroying the last VkInstance in
     * the process tears down driver state the EGL dma-buf import path is
     * still using, and every later framebuffer over an EGLImage comes
     * back incomplete. Dawn keeps an instance alive for its own reasons,
     * which is why a gl -> webgpu run never saw this and a gl -> cpu run
     * did. So this instance outlives the call on purpose. */
    static VkInstance inst = VK_NULL_HANDLE;
    static uint64_t cache[64];
    static int cached = -1;
    if (cached >= 0) {
        int n = cached < max ? cached : max;
        memcpy(out, cache, (size_t)n * sizeof *out);
        return n;
    }
    cached = 0;
    VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &ai};
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { return 0; }
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, NULL);
    int n = 0;
    if (count) {
        VkPhysicalDevice phys = VK_NULL_HANDLE;
        count = 1;
        vkEnumeratePhysicalDevices(inst, &count, &phys);
        VkDrmFormatModifierPropertiesEXT props[64];
        VkDrmFormatModifierPropertiesListEXT list = {
            .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
            .drmFormatModifierCount = 64,
            .pDrmFormatModifierProperties = props};
        VkFormatProperties2 fp = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &list};
        vkGetPhysicalDeviceFormatProperties2(phys, VK_FORMAT_R8G8B8A8_UNORM, &fp);
        for (uint32_t i = 0; i < list.drmFormatModifierCount && n < max; i++) {
            /* single-plane only: the tensor is one image, and the edge
             * describes it with one fd, offset and pitch */
            if (props[i].drmFormatModifierPlaneCount == 1) {
                out[n++] = props[i].drmFormatModifier;
            }
        }
    }
    cached = n;
    memcpy(cache, out, (size_t)n * sizeof *out);
    return n;
}
