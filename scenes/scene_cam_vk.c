/* scene_cam_vk.c -- the camera through Vulkan.
 *
 * Two ways in, chosen at runtime by what the driver offers:
 *
 *   ZERO-COPY (dmabuf import). Each camera buffer becomes a VkImage
 *   whose memory is IMPORTED from the dmabuf fd instead of allocated:
 *   VkExternalMemoryImageCreateInfo + VkImageDrmFormatModifierExplicit-
 *   CreateInfoEXT describe the layout (linear, two planes at the
 *   offsets V4L2 reported), VkImportMemoryFdInfoKHR wraps the fd. The
 *   format is the 2-plane NV12 format and a VkSamplerYcbcrConversion
 *   does YUV->RGB inside the sampler -- the shader sees a sampler2D.
 *   Ownership is passed with queue-family barriers to/from
 *   VK_QUEUE_FAMILY_FOREIGN_EXT ("not a Vulkan queue": the ISP).
 *
 *   FALLBACK (staging). The CPU converts NV12 -> RGBA into a
 *   host-visible buffer, vkCmdCopyBufferToImage moves it into a
 *   device-local image. The classic upload path every texture takes.
 *
 * Both need the machinery push constants avoided so far: a descriptor
 * set layout (what bindings the shader has), a pool, a set per camera
 * image, vkCmdBindDescriptorSets.
 *
 * The effect chain (effect_chain.h) is a sequence of rendering
 * instances ping-ponging between two scratch images -- each a color
 * attachment while written, a sampled texture while read, with a
 * layout barrier at every switch. The last pass renders to the
 * window, or (C) onto the cube.
 *
 * Buffer lifetime: the frame serial from vk_frame is the "fence"; a
 * buffer returns to V4L2 once vk_present_serial_done says the frame
 * that sampled it has finished.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <drm_fourcc.h>

#include "cam_stream.h"
#include "cube_mesh.h"
#include "effect_chain.h"
#include "mat4.h"
#include "nv12_convert.h"
#include "vk_present.h"

#ifndef SHADER_DIR
#define SHADER_DIR "./shaders"
#endif

/* The cube pipeline's push constants: the vertex stage gets the mvp
 * and the light direction pre-rotated into MODEL space (80 bytes),
 * the fragment stage gets the same pass parameters as the flat
 * pipeline (struct cam_push, at offset 96). 116 bytes total: the 128
 * guaranteed minimum is why the model matrix is not pushed too --
 * lighting a flat-shaded cube needs only normal . light, which works
 * in any space as long as both are in the same one. */
struct cube_push { /* must match cube_cam.vert */
    mat4 mvp;
    float light_model[4];
    float uv_scale[2]; /* aspect: center-crop the camera onto the square face */
    float pad[2];
};
enum { CUBE_FRAG_PUSH_OFFSET = sizeof(struct cube_push) };

struct cam_push { /* must match cam.frag / cube_cam.frag */
    float texel[2];
    float time;
    int32_t effect;
    int32_t src; /* 0 camera, 1 scratch A, 2 scratch B */
};

struct cam_scene {
    struct cam_stream stream;
    uint32_t cw, ch;
    int zero_copy;

    /* the camera as Vulkan images: one per dmabuf, or one RGBA target */
    uint32_t cam_image_count;
    VkImage cam_image[CAM_MAX_BUFFERS];
    VkDeviceMemory cam_mem[CAM_MAX_BUFFERS];
    VkImageView cam_view[CAM_MAX_BUFFERS];
    VkSamplerYcbcrConversion ycbcr; /* zero-copy only */
    VkSampler cam_sampler;
    PFN_vkGetMemoryFdPropertiesKHR get_fd_props;

    /* fallback: per-lane staging + a CPU conversion scratch */
    VkBuffer staging[VK_PRESENT_LANES];
    VkDeviceMemory staging_mem[VK_PRESENT_LANES];
    void* staging_map[VK_PRESENT_LANES];
    uint32_t* rgb_scratch;
    int fallback_image_written;

    /* the chain's scratch: attachment while written, texture while read */
    VkImage tmp_image[2];
    VkDeviceMemory tmp_mem[2];
    VkImageView tmp_view[2];
    VkSampler tmp_sampler;

    VkDescriptorSetLayout set_layout;
    VkDescriptorPool pool;
    VkDescriptorSet sets[CAM_MAX_BUFFERS];
    VkPipelineLayout layout;
    VkPipeline pipeline;

    /* the cube as the last pass: scene_cube_vk.c's pipeline with a uv
     * attribute, sampling through the same descriptor set */
    VkBuffer vbo, ibo;
    VkDeviceMemory vbo_mem, ibo_mem;
    VkPipelineLayout cube_layout;
    VkPipeline cube_pipeline;
};

/* ------------------------------------------------------------------ */
/* fences: frame serials                                              */
/* ------------------------------------------------------------------ */

static int serial_done(void* fence, void* user) {
    return vk_present_serial_done(user, (uint64_t)(uintptr_t)fence);
}

static const struct cam_fence_ops serial_fence_ops = {.done = serial_done, .destroy = NULL};

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

static void create_image(struct vk_presenter* p,
                         uint32_t w,
                         uint32_t h,
                         VkFormat format,
                         VkImageUsageFlags usage,
                         VkImage* image,
                         VkDeviceMemory* mem) {
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VK_CHECK(vkCreateImage(p->device, &ici, NULL, image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(p->device, *image, &req);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex =
            vk_present_find_memory_type(p, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    VK_CHECK(vkAllocateMemory(p->device, &mai, NULL, mem));
    VK_CHECK(vkBindImageMemory(p->device, *image, *mem, 0));
}

static VkImageView create_view(struct vk_presenter* p,
                               VkImage image,
                               VkFormat format,
                               const void* pnext) {
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = pnext,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageView view;
    VK_CHECK(vkCreateImageView(p->device, &vci, NULL, &view));
    return view;
}

static VkSampler create_sampler(struct vk_presenter* p, const void* pnext, VkFilter filter) {
    VkSamplerCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = pnext,
        .magFilter = filter,
        .minFilter = filter,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = 0.0f,
    };
    VkSampler s;
    VK_CHECK(vkCreateSampler(p->device, &sci, NULL, &s));
    return s;
}

static void image_barrier(VkCommandBuffer cmd,
                          VkImage image,
                          VkImageLayout old_layout,
                          VkImageLayout new_layout,
                          VkPipelineStageFlags src_stage,
                          VkAccessFlags src_access,
                          VkPipelineStageFlags dst_stage,
                          VkAccessFlags dst_access,
                          uint32_t src_family,
                          uint32_t dst_family) {
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = src_family,
        .dstQueueFamilyIndex = dst_family,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

/* ------------------------------------------------------------------ */
/* zero-copy import                                                   */
/* ------------------------------------------------------------------ */

static int import_dmabufs(struct vk_presenter* p, struct cam_scene* s) {
    const struct camera* cam = &s->stream.cam;
    if (cam->format.pixelformat != V4L2_PIX_FMT_NV12) { return -1; }
    const VkFormat fmt = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM; /* NV12 */

    s->get_fd_props =
        (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(p->device,
                                                            "vkGetMemoryFdPropertiesKHR");
    if (!s->get_fd_props) { return -1; }

    /* What can the sampler do with this format? Chroma is at half
     * resolution; where exactly the sample sits (cosited/midpoint) and
     * whether it may be filtered are format features. */
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(p->phys, fmt, &fp);
    VkFormatFeatureFlags feat = fp.linearTilingFeatures;
    if (!(feat & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
        fprintf(stderr, "cam_vk: NV12 not sampleable with linear tiling\n");
        return -1;
    }
    VkChromaLocation chroma = (feat & VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT)
                                  ? VK_CHROMA_LOCATION_COSITED_EVEN
                                  : VK_CHROMA_LOCATION_MIDPOINT;
    VkFilter chroma_filter =
        (feat & VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT)
            ? VK_FILTER_LINEAR
            : VK_FILTER_NEAREST;

    /* The conversion object: which matrix, which range, chroma siting.
     * The GPU's texture unit applies it on every sample. */
    VkSamplerYcbcrConversionCreateInfo yci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
        .format = fmt,
        .ycbcrModel = camera_is_bt709(cam) ? VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709
                                           : VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601,
        .ycbcrRange = camera_is_full_range(cam) ? VK_SAMPLER_YCBCR_RANGE_ITU_FULL
                                                : VK_SAMPLER_YCBCR_RANGE_ITU_NARROW,
        .components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                       VK_COMPONENT_SWIZZLE_IDENTITY,
                       VK_COMPONENT_SWIZZLE_IDENTITY,
                       VK_COMPONENT_SWIZZLE_IDENTITY},
        .xChromaOffset = chroma,
        .yChromaOffset = chroma,
        .chromaFilter = chroma_filter,
    };
    VK_CHECK(vkCreateSamplerYcbcrConversion(p->device, &yci, NULL, &s->ycbcr));
    VkSamplerYcbcrConversionInfo yinfo = {.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
                                          .conversion = s->ycbcr};
    s->cam_sampler = create_sampler(p, &yinfo, VK_FILTER_LINEAR);

    for (uint32_t i = 0; i < cam->buf_count; i++) {
        /* 1. an image whose memory will come from outside, with an
         *    explicitly described linear layout: the two planes at the
         *    offsets/pitch V4L2 gave us */
        VkSubresourceLayout planes[2] = {
            {.offset = 0, .rowPitch = camera_stride(cam)},
            {.offset = camera_uv_offset(cam), .rowPitch = camera_stride(cam)},
        };
        VkImageDrmFormatModifierExplicitCreateInfoEXT modinfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
            .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
            .drmFormatModifierPlaneCount = 2,
            .pPlaneLayouts = planes,
        };
        VkExternalMemoryImageCreateInfo extinfo = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            .pNext = &modinfo,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        };
        VkImageCreateInfo ici = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &extinfo,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = fmt,
            .extent = {cam->format.width, cam->format.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        if (vkCreateImage(p->device, &ici, NULL, &s->cam_image[i]) != VK_SUCCESS) {
            fprintf(stderr, "cam_vk: vkCreateImage(dmabuf NV12) failed\n");
            return -1;
        }

        /* 2. the memory: not allocated, IMPORTED. The driver tells us
         *    which memory types can back this fd; the fd is dup'd
         *    because a successful import consumes it. */
        int fd = cam->bufs[i].dmabuf_fd;
        VkMemoryFdPropertiesKHR fdp = {.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
        VK_CHECK(
            s->get_fd_props(p->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fdp));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(p->device, s->cam_image[i], &req);
        uint32_t bits = req.memoryTypeBits & fdp.memoryTypeBits;
        if (!bits) {
            fprintf(stderr, "cam_vk: no memory type can import the dmabuf\n");
            return -1;
        }
        VkMemoryDedicatedAllocateInfo dedicated = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
            .image = s->cam_image[i],
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
            .allocationSize = req.size,
            .memoryTypeIndex = vk_present_find_memory_type(p, bits, 0),
        };
        if (vkAllocateMemory(p->device, &mai, NULL, &s->cam_mem[i]) != VK_SUCCESS) {
            fprintf(stderr, "cam_vk: dmabuf import failed\n");
            close(import.fd);
            return -1;
        }
        VK_CHECK(vkBindImageMemory(p->device, s->cam_image[i], s->cam_mem[i], 0));

        /* 3. a view that carries the conversion */
        s->cam_view[i] = create_view(p, s->cam_image[i], fmt, &yinfo);
    }
    s->cam_image_count = cam->buf_count;
    fprintf(stderr,
            "cam_vk: %u camera dmabufs imported (NV12, %s, %s)\n",
            cam->buf_count,
            camera_is_bt709(cam) ? "BT.709" : "BT.601",
            camera_is_full_range(cam) ? "full range" : "limited range");
    return 0;
}

/* ------------------------------------------------------------------ */
/* fallback: staging upload                                           */
/* ------------------------------------------------------------------ */

static void create_fallback(struct vk_presenter* p, struct cam_scene* s) {
    create_image(p,
                 s->cw,
                 s->ch,
                 VK_FORMAT_B8G8R8A8_UNORM,
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                 &s->cam_image[0],
                 &s->cam_mem[0]);
    s->cam_view[0] = create_view(p, s->cam_image[0], VK_FORMAT_B8G8R8A8_UNORM, NULL);
    s->cam_sampler = create_sampler(p, NULL, VK_FILTER_LINEAR);
    s->cam_image_count = 1;

    VkDeviceSize bytes = (VkDeviceSize)s->cw * s->ch * 4;
    for (int i = 0; i < VK_PRESENT_LANES; i++) {
        VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .size = bytes,
                                  .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VK_CHECK(vkCreateBuffer(p->device, &bci, NULL, &s->staging[i]));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(p->device, s->staging[i], &req);
        VkMemoryAllocateInfo mai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = req.size,
            .memoryTypeIndex = vk_present_find_memory_type(
                p,
                req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        };
        VK_CHECK(vkAllocateMemory(p->device, &mai, NULL, &s->staging_mem[i]));
        VK_CHECK(vkBindBufferMemory(p->device, s->staging[i], s->staging_mem[i], 0));
        VK_CHECK(vkMapMemory(p->device, s->staging_mem[i], 0, bytes, 0, &s->staging_map[i]));
    }
    s->rgb_scratch = malloc((size_t)bytes);
}

/* ------------------------------------------------------------------ */
/* descriptors + pipelines                                            */
/* ------------------------------------------------------------------ */

static void create_descriptors(struct vk_presenter* p, struct cam_scene* s) {
    /* The layout: what the shader's `set = 0` contains. Binding 0 is
     * the camera with an IMMUTABLE sampler -- required for YCbCr
     * conversion samplers, and reasonable anyway: the sampler is part
     * of the pipeline's contract, not per-frame data. 1 and 2 are the
     * chain's scratch textures. */
    VkDescriptorSetLayoutBinding bindings[3] = {
        {.binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
         .pImmutableSamplers = &s->cam_sampler},
        {.binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
         .pImmutableSamplers = &s->tmp_sampler},
        {.binding = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
         .pImmutableSamplers = &s->tmp_sampler},
    };
    VkDescriptorSetLayoutCreateInfo lci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(p->device, &lci, NULL, &s->set_layout));

    /* The pool: descriptor memory, sized up front. */
    VkDescriptorPoolSize size = {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 .descriptorCount = 3 * CAM_MAX_BUFFERS};
    VkDescriptorPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = CAM_MAX_BUFFERS,
        .poolSizeCount = 1,
        .pPoolSizes = &size,
    };
    VK_CHECK(vkCreateDescriptorPool(p->device, &pci, NULL, &s->pool));

    /* One set per camera image, each pointing binding 0 at its image
     * and 1/2 at the shared scratch. Written once; per frame we only
     * pick which set to bind and which binding a pass reads (push). */
    for (uint32_t i = 0; i < s->cam_image_count; i++) {
        VkDescriptorSetAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = s->pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &s->set_layout,
        };
        VK_CHECK(vkAllocateDescriptorSets(p->device, &ai, &s->sets[i]));
        VkDescriptorImageInfo infos[3] = {
            {.imageView = s->cam_view[i], .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {.imageView = s->tmp_view[0], .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {.imageView = s->tmp_view[1], .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        };
        VkWriteDescriptorSet writes[3];
        for (uint32_t b = 0; b < 3; b++) {
            writes[b] = (VkWriteDescriptorSet){
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = s->sets[i],
                .dstBinding = b,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &infos[b],
            };
        }
        vkUpdateDescriptorSets(p->device, 3, writes, 0, NULL);
    }
}

static void create_pipeline(struct vk_presenter* p, struct cam_scene* s) {
    VkShaderModule vs = vk_present_load_shader(p, SHADER_DIR "/cam.vert.spv");
    VkShaderModule fs = vk_present_load_shader(p, SHADER_DIR "/cam.frag.spv");

    VkPushConstantRange pc_range = {.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                                    .offset = 0,
                                    .size = sizeof(struct cam_push)};
    VkPipelineLayoutCreateInfo lci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, /* the descriptor channel, finally in use */
        .pSetLayouts = &s->set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc_range,
    };
    VK_CHECK(vkCreatePipelineLayout(p->device, &lci, NULL, &s->layout));

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
    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dyn_states};
    /* the scratch images use the swapchain's format on purpose: one
     * pipeline serves every pass */
    VkPipelineRenderingCreateInfo rendering = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &p->color_format,
    };
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
        .layout = s->layout,
    };
    VK_CHECK(vkCreateGraphicsPipelines(p->device, VK_NULL_HANDLE, 1, &pci, NULL, &s->pipeline));
    vkDestroyShaderModule(p->device, vs, NULL);
    vkDestroyShaderModule(p->device, fs, NULL);
}

static void create_host_buffer(struct vk_presenter* p,
                               VkBufferUsageFlags usage,
                               const void* data,
                               VkDeviceSize size,
                               VkBuffer* buf,
                               VkDeviceMemory* mem) {
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = size,
                              .usage = usage,
                              .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VK_CHECK(vkCreateBuffer(p->device, &bci, NULL, buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(p->device, *buf, &req);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = vk_present_find_memory_type(
            p,
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };
    VK_CHECK(vkAllocateMemory(p->device, &mai, NULL, mem));
    VK_CHECK(vkBindBufferMemory(p->device, *buf, *mem, 0));
    void* mapped;
    VK_CHECK(vkMapMemory(p->device, *mem, 0, size, 0, &mapped));
    memcpy(mapped, data, (size_t)size);
    vkUnmapMemory(p->device, *mem);
}

static void create_cube_pipeline(struct vk_presenter* p, struct cam_scene* s) {
    create_host_buffer(p,
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                       cube_vertices,
                       sizeof(cube_vertices),
                       &s->vbo,
                       &s->vbo_mem);
    create_host_buffer(p,
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                       cube_indices,
                       sizeof(cube_indices),
                       &s->ibo,
                       &s->ibo_mem);

    VkShaderModule vs = vk_present_load_shader(p, SHADER_DIR "/cube_cam.vert.spv");
    VkShaderModule fs = vk_present_load_shader(p, SHADER_DIR "/cube_cam.frag.spv");

    /* same descriptor set layout (so the same sets bind); two push
     * ranges, one per stage, back to back */
    VkPushConstantRange pc_ranges[2] = {
        {.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(struct cube_push)},
        {.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
         .offset = CUBE_FRAG_PUSH_OFFSET,
         .size = sizeof(struct cam_push)},
    };
    VkPipelineLayoutCreateInfo lci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &s->set_layout,
        .pushConstantRangeCount = 2,
        .pPushConstantRanges = pc_ranges,
    };
    VK_CHECK(vkCreatePipelineLayout(p->device, &lci, NULL, &s->cube_layout));

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
    VkVertexInputBindingDescription binding = {.binding = 0,
                                               .stride = sizeof(struct cube_vertex),
                                               .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[4] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(struct cube_vertex, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(struct cube_vertex, normal)},
        {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(struct cube_vertex, color)},
        {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(struct cube_vertex, uv)},
    };
    VkPipelineVertexInputStateCreateInfo vin = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 4,
        .pVertexAttributeDescriptions = attrs,
    };
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
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };
    VkPipelineColorBlendAttachmentState blend_att = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    VkPipelineColorBlendStateCreateInfo blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blend_att};
    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dyn_states};
    VkPipelineRenderingCreateInfo rendering = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &p->color_format,
        .depthAttachmentFormat = p->depth_format,
    };
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
        .pDepthStencilState = &ds,
        .pColorBlendState = &blend,
        .pDynamicState = &dyn,
        .layout = s->cube_layout,
    };
    VK_CHECK(
        vkCreateGraphicsPipelines(p->device, VK_NULL_HANDLE, 1, &pci, NULL, &s->cube_pipeline));
    vkDestroyShaderModule(p->device, vs, NULL);
    vkDestroyShaderModule(p->device, fs, NULL);
}

/* ------------------------------------------------------------------ */

static void on_cam_fd(struct app* a) {
    struct vk_presenter* p = (struct vk_presenter*)a;
    struct cam_scene* s = p->user;
    cam_stream_pump(&s->stream);
}

static void cam_init(struct vk_presenter* p, void* user) {
    struct cam_scene* s = user;
    if (cam_stream_open(&s->stream, &serial_fence_ops, p) < 0) { exit(1); }
    s->cw = s->stream.cam.format.width;
    s->ch = s->stream.cam.format.height;

    s->zero_copy = p->dmabuf_import && import_dmabufs(p, s) == 0;
    if (!s->zero_copy) {
        fprintf(stderr, "cam_vk: using the CPU staging path\n");
        create_fallback(p, s);
    }

    s->tmp_sampler = create_sampler(p, NULL, VK_FILTER_LINEAR);
    for (int i = 0; i < 2; i++) {
        create_image(p,
                     s->cw,
                     s->ch,
                     p->color_format,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                     &s->tmp_image[i],
                     &s->tmp_mem[i]);
        s->tmp_view[i] = create_view(p, s->tmp_image[i], p->color_format, NULL);
    }

    create_descriptors(p, s);
    create_pipeline(p, s);
    create_cube_pipeline(p, s);

    p->app.aux_fd = cam_stream_fd(&s->stream);
    p->app.on_aux_fd = on_cam_fd;
    p->app.effect_count = FX_COUNT;
}

/* One fullscreen pass with the flat pipeline into whatever rendering
 * instance is open, over `rect`. */
static void draw_fullscreen(struct cam_scene* s,
                            VkCommandBuffer cmd,
                            VkDescriptorSet set,
                            const struct cam_push* push,
                            VkRect2D rect) {
    VkViewport viewport = {(float)rect.offset.x,
                           (float)rect.offset.y,
                           (float)rect.extent.width,
                           (float)rect.extent.height,
                           0.0f,
                           1.0f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &rect);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cmd, s->layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(*push), push);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

/* The cube as the chain's last pass: its own rendering instance with
 * color + depth. */
static void draw_cube(struct vk_presenter* p,
                      struct cam_scene* s,
                      const struct vk_frame* fr,
                      VkDescriptorSet set,
                      const struct cam_push* fx) {
    VkRenderingAttachmentInfo color_att = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = fr->color,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.color = {{0.08f, 0.08f, 0.10f, 1.0f}}},
    };
    VkRenderingAttachmentInfo depth_att = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = fr->depth,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .clearValue = {.depthStencil = {1.0f, 0}},
    };
    VkRenderingInfo ri = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, fr->extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_att,
        .pDepthAttachment = &depth_att,
    };
    vkCmdBeginRendering(fr->cmd, &ri);
    VkViewport viewport = {0, 0, (float)fr->extent.width, (float)fr->extent.height, 0.0f, 1.0f};
    VkRect2D scissor = {{0, 0}, fr->extent};
    vkCmdSetViewport(fr->cmd, 0, 1, &viewport);
    vkCmdSetScissor(fr->cmd, 0, 1, &scissor);
    vkCmdBindPipeline(fr->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->cube_pipeline);
    vkCmdBindDescriptorSets(fr->cmd,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            s->cube_layout,
                            0,
                            1,
                            &set,
                            0,
                            NULL);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fr->cmd, 0, 1, &s->vbo, &offset);
    vkCmdBindIndexBuffer(fr->cmd, s->ibo, 0, VK_INDEX_TYPE_UINT16);

    mat4 model = cube_model_matrix(&p->app);
    struct cube_push push;
    push.mvp =
        mat4_mul(cube_view_proj((int)fr->extent.width, (int)fr->extent.height, 1, 1, p->app.zoom),
                 model);
    /* the light, rotated INTO model space: R^T * light, i.e. the dot of
     * each rotation column with the light vector (R is orthonormal) */
    static const float light[3] = {0.4f / 1.3416408f, 0.8f / 1.3416408f, 1.0f / 1.3416408f};
    for (int i = 0; i < 3; i++) {
        push.light_model[i] = model.m[i * 4 + 0] * light[0] + model.m[i * 4 + 1] * light[1] +
                              model.m[i * 4 + 2] * light[2];
    }
    push.light_model[3] = 0.0f;
    /* "cover": full extent of the camera's short axis, a centered
     * window of the long one */
    const float aspect = (float)s->cw / (float)s->ch;
    push.uv_scale[0] = aspect > 1.0f ? 1.0f / aspect : 1.0f;
    push.uv_scale[1] = aspect > 1.0f ? 1.0f : aspect;
    push.pad[0] = push.pad[1] = 0.0f;
    vkCmdPushConstants(fr->cmd, s->cube_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
    vkCmdPushConstants(fr->cmd,
                       s->cube_layout,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       CUBE_FRAG_PUSH_OFFSET,
                       sizeof(*fx),
                       fx);
    vkCmdDrawIndexed(fr->cmd, CUBE_INDEX_COUNT, 1, 0, 0, 0);
    vkCmdEndRendering(fr->cmd);
}

static void cam_record(struct vk_presenter* p, const struct vk_frame* fr, void* user) {
    struct cam_scene* s = user;
    VkCommandBuffer cmd = fr->cmd;
    cam_stream_reap(&s->stream);

    int latest = cam_stream_latest(&s->stream);
    int have = latest >= 0;
    uint32_t set_index = 0;

    if (have && s->zero_copy) {
        set_index = (uint32_t)latest;
        /* acquire from the foreign owner (the ISP): from now on the
         * graphics queue reads it */
        image_barrier(cmd,
                      s->cam_image[latest],
                      VK_IMAGE_LAYOUT_GENERAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      0,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT,
                      VK_QUEUE_FAMILY_FOREIGN_EXT,
                      p->queue_family);
    } else if (have) {
        /* staging: CPU converts into this lane's buffer, GPU copies it
         * into the sampled image */
        const struct camera* cam = &s->stream.cam;
        const uint8_t* base = camera_map(&s->stream.cam, (uint32_t)latest);
        nv12_to_xrgb(base,
                     base + camera_uv_offset(cam),
                     camera_stride(cam),
                     s->cw,
                     s->ch,
                     camera_is_bt709(cam),
                     camera_is_full_range(cam),
                     s->rgb_scratch);
        memcpy(s->staging_map[fr->lane], s->rgb_scratch, (size_t)s->cw * s->ch * 4);

        image_barrier(cmd,
                      s->cam_image[0],
                      s->fallback_image_written ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                : VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_QUEUE_FAMILY_IGNORED,
                      VK_QUEUE_FAMILY_IGNORED);
        VkBufferImageCopy region = {
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {s->cw, s->ch, 1},
        };
        vkCmdCopyBufferToImage(cmd,
                               s->staging[fr->lane],
                               s->cam_image[0],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1,
                               &region);
        image_barrier(cmd,
                      s->cam_image[0],
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT,
                      VK_QUEUE_FAMILY_IGNORED,
                      VK_QUEUE_FAMILY_IGNORED);
        s->fallback_image_written = 1;
    }

    struct cam_push push = {
        .texel = {1.0f / (float)s->cw, 1.0f / (float)s->ch},
        .time = (float)(p->app.anim_ms * 0.001),
        .effect = PASS_NONE,
        .src = 0,
    };

    /* The chain. Pass i reads stage push.src (0 camera, 1/2 scratch
     * A/B) and writes scratch i&1 -- a rendering instance per pass,
     * with the scratch flipped to an attachment before and back to a
     * texture after. The last pass goes to the window, or the cube. */
    int passes[EFFECT_MAX_PASSES];
    int n = have ? effect_chain_passes(p->app.effect, passes) : 0;
    for (int i = 0; i + 1 < n; i++) {
        const int d = i & 1;
        push.effect = passes[i];
        image_barrier(cmd,
                      s->tmp_image[d],
                      VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                      VK_QUEUE_FAMILY_IGNORED,
                      VK_QUEUE_FAMILY_IGNORED);
        VkRenderingAttachmentInfo att = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = s->tmp_view[d],
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        };
        VkRenderingInfo ri = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{0, 0}, {s->cw, s->ch}},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &att,
        };
        vkCmdBeginRendering(cmd, &ri);
        draw_fullscreen(s, cmd, s->sets[set_index], &push, (VkRect2D){{0, 0}, {s->cw, s->ch}});
        vkCmdEndRendering(cmd);
        image_barrier(cmd,
                      s->tmp_image[d],
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT,
                      VK_QUEUE_FAMILY_IGNORED,
                      VK_QUEUE_FAMILY_IGNORED);
        push.src = 1 + d;
    }
    if (n > 0) { push.effect = passes[n - 1]; }

    if (have && p->app.mode) {
        draw_cube(p, s, fr, s->sets[set_index], &push);
    } else {
        /* the window: clear to black, draw the letterboxed camera */
        VkRenderingAttachmentInfo color_att = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = fr->color,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.color = {{0.0f, 0.0f, 0.0f, 1.0f}}},
        };
        VkRenderingInfo ri = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{0, 0}, fr->extent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_att,
        };
        vkCmdBeginRendering(cmd, &ri);
        if (have) {
            float sx = (float)fr->extent.width / (float)s->cw;
            float sy = (float)fr->extent.height / (float)s->ch;
            float sc = sx < sy ? sx : sy;
            uint32_t dw = (uint32_t)((float)s->cw * sc), dh = (uint32_t)((float)s->ch * sc);
            VkRect2D rect = {
                {(int32_t)(fr->extent.width - dw) / 2, (int32_t)(fr->extent.height - dh) / 2},
                {dw, dh}};
            draw_fullscreen(s, cmd, s->sets[set_index], &push, rect);
        }
        vkCmdEndRendering(cmd);
    }

    if (have && s->zero_copy) {
        /* release back to the foreign owner */
        image_barrier(cmd,
                      s->cam_image[latest],
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      0,
                      p->queue_family,
                      VK_QUEUE_FAMILY_FOREIGN_EXT);
    }
    if (have) { cam_stream_rendered(&s->stream, latest, (void*)(uintptr_t)fr->serial); }
}

static void cam_fini(struct vk_presenter* p, void* user) {
    struct cam_scene* s = user;
    cam_stream_close(&s->stream);
    vkDestroyPipeline(p->device, s->cube_pipeline, NULL);
    vkDestroyPipelineLayout(p->device, s->cube_layout, NULL);
    vkDestroyBuffer(p->device, s->vbo, NULL);
    vkFreeMemory(p->device, s->vbo_mem, NULL);
    vkDestroyBuffer(p->device, s->ibo, NULL);
    vkFreeMemory(p->device, s->ibo_mem, NULL);
    vkDestroyPipeline(p->device, s->pipeline, NULL);
    vkDestroyPipelineLayout(p->device, s->layout, NULL);
    vkDestroyDescriptorPool(p->device, s->pool, NULL);
    vkDestroyDescriptorSetLayout(p->device, s->set_layout, NULL);
    vkDestroySampler(p->device, s->tmp_sampler, NULL);
    for (int i = 0; i < 2; i++) {
        vkDestroyImageView(p->device, s->tmp_view[i], NULL);
        vkDestroyImage(p->device, s->tmp_image[i], NULL);
        vkFreeMemory(p->device, s->tmp_mem[i], NULL);
    }
    for (uint32_t i = 0; i < s->cam_image_count; i++) {
        vkDestroyImageView(p->device, s->cam_view[i], NULL);
        vkDestroyImage(p->device, s->cam_image[i], NULL);
        vkFreeMemory(p->device, s->cam_mem[i], NULL);
    }
    vkDestroySampler(p->device, s->cam_sampler, NULL);
    if (s->ycbcr) { vkDestroySamplerYcbcrConversion(p->device, s->ycbcr, NULL); }
    for (int i = 0; i < VK_PRESENT_LANES; i++) {
        if (s->staging[i]) {
            vkDestroyBuffer(p->device, s->staging[i], NULL);
            vkFreeMemory(p->device, s->staging_mem[i], NULL);
        }
    }
    free(s->rgb_scratch);
}

static const struct vk_scene cam_scene = {
    .init = cam_init,
    .resize = NULL,
    .record = cam_record,
    .fini = cam_fini,
};

int main(void) {
    struct cam_scene s = {0};
    struct vk_presenter p;
    if (vk_present_init(&p,
                        VK_PRESENT_CAMERA_IMPORT | VK_PRESENT_DEPTH,
                        &cam_scene,
                        &s,
                        "camera (Vulkan)",
                        "hello-wayland-cam-vulkan") < 0) {
        return 1;
    }
    vk_present_run(&p);
    vk_present_fini(&p);
    return 0;
}
