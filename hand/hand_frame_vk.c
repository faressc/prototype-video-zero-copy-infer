/* hand_frame_vk.c -- FrameToTensor on the headless Vulkan device: the
 * camera's dma-buf planes (infer_vk_frame_import), one compute pass
 * (shaders/hand_frame_to_tensor.comp, SPIR-V at build time), a packed
 * float SSBO out. The third transcription of the same formula as
 * hand_frame_cpu.c and hand_frame_to_tensor.wgsl, held to the same
 * agreement bar by hello_hand --compare-frame-to-tensor.
 *
 * Whose buffer the pass writes is the caller's business: a VK tensor's
 * buffer alias (the harness; the vk -> cuda edge's source), or the
 * opaque-fd buffer the CUDA EP reads through its mapping -- in which
 * case `signal` carries the semaphore the CUDA stream will wait on, and
 * the feed is one device pass with no host bytes, the WebGPU
 * arrangement with Vulkan in Dawn's seat.
 *
 * Serialisation is the pass's own fence, waited at the START of the next
 * apply (and by hand_frame_vk_wait), never busily: the caller's cycle is
 * synchronous -- run, then sync, then the next feed -- so by the time a
 * new frame is fed, everything reading the previous tensor has finished.
 * The submit's completion can leave as a SYNC_FD (`release_fd`): the
 * "done reading the frame" fence the camera's buffer lifetime wants,
 * the same contract infer_wgpu_frame_end exports.
 */
#include <stdlib.h>
#include <string.h>

#include <drm_fourcc.h>

#include "hand_frame.h"
#include "infer_util.h"
#include "nv12_convert.h"

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

/* mirrors the push-constant block in hand_frame_to_tensor.comp: six
 * vec4s, 96 bytes, inside the 128-byte minimum */
struct pass_params {
    float affine[4]; /* m0 m1 m3 m4 */
    float extra[4];  /* m2, m5, norm lo, norm span */
    uint32_t sizes[4];
    float coef[4];
    float range[4];    /* y off, y scale, chroma scale, unused */
    uint32_t flags[4]; /* yuyv, border_replicate, unused, unused */
};

struct hand_frame_pass_vk {
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    VkDescriptorPool dpool;
    VkDescriptorSet set;
    VkSampler sampler; /* texelFetch never consults it; the binding needs one */
    VkCommandBuffer cmd; /* our own: c->cmd belongs to the synchronous paths */
    VkFence fence;
    int fence_armed;
    VkFence armed_fence; /* ours, or the tensor's when writing a VK tensor */
    VkSemaphore release_sem; /* SYNC_FD out, when the driver can */
    int has_release;
};

struct hand_frame_pass_vk* hand_frame_pass_vk_create(struct infer_ctx_vk* c) {
    if (!c->device) { return NULL; }
    struct hand_frame_pass_vk* p = calloc(1, sizeof *p);
    if (!p) { return NULL; }

    VkDescriptorSetLayoutBinding b[3] = {
        {.binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo lci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = b,
    };
    if (vkCreateDescriptorSetLayout(c->device, &lci, NULL, &p->dsl) != VK_SUCCESS) { goto fail; }
    VkPushConstantRange pc = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                              .size = sizeof(struct pass_params)};
    VkPipelineLayoutCreateInfo pl = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &p->dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc,
    };
    if (vkCreatePipelineLayout(c->device, &pl, NULL, &p->layout) != VK_SUCCESS) { goto fail; }

    size_t size = 0;
    void* code = infer_read_file(SHADER_DIR "/hand_frame_to_tensor.comp.spv", &size);
    if (!code) { goto fail; }
    VkShaderModuleCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = size,
                                    .pCode = code};
    VkShaderModule mod;
    VkResult r = vkCreateShaderModule(c->device, &sci, NULL, &mod);
    free(code);
    if (r != VK_SUCCESS) { goto fail; }
    VkComputePipelineCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = mod,
                  .pName = "main"},
        .layout = p->layout,
    };
    r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &pci, NULL, &p->pipeline);
    vkDestroyShaderModule(c->device, mod, NULL);
    if (r != VK_SUCCESS) { goto fail; }

    VkSamplerCreateInfo smi = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };
    if (vkCreateSampler(c->device, &smi, NULL, &p->sampler) != VK_SUCCESS) { goto fail; }

    VkDescriptorPoolSize sizes[2] = {
        {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 2},
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1},
    };
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                      .maxSets = 1,
                                      .poolSizeCount = 2,
                                      .pPoolSizes = sizes};
    if (vkCreateDescriptorPool(c->device, &dpi, NULL, &p->dpool) != VK_SUCCESS) { goto fail; }
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                       .descriptorPool = p->dpool,
                                       .descriptorSetCount = 1,
                                       .pSetLayouts = &p->dsl};
    if (vkAllocateDescriptorSets(c->device, &dsa, &p->set) != VK_SUCCESS) { goto fail; }

    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = c->cmd_pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1};
    if (vkAllocateCommandBuffers(c->device, &cai, &p->cmd) != VK_SUCCESS) { goto fail; }
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(c->device, &fci, NULL, &p->fence) != VK_SUCCESS) { goto fail; }
    if (c->has_sync_fd) {
        VkExportSemaphoreCreateInfo esi = {
            .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
        VkSemaphoreCreateInfo semci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                       .pNext = &esi};
        if (vkCreateSemaphore(c->device, &semci, NULL, &p->release_sem) == VK_SUCCESS) {
            p->has_release = 1;
        }
    }
    return p;

fail:
    fprintf(stderr, "hand: vk frame_to_tensor pipeline creation failed\n");
    hand_frame_pass_vk_destroy(c, p);
    return NULL;
}

int hand_frame_vk_wait(struct infer_ctx_vk* c, struct hand_frame_pass_vk* p) {
    if (!p || !p->fence_armed) { return 0; }
    VkFence f = p->armed_fence ? p->armed_fence : p->fence;
    VK_CHECK(vkWaitForFences(c->device, 1, &f, VK_TRUE, UINT64_MAX));
    p->fence_armed = 0;
    return 0;
}

void hand_frame_pass_vk_destroy(struct infer_ctx_vk* c, struct hand_frame_pass_vk* p) {
    if (!p) { return; }
    hand_frame_vk_wait(c, p);
    if (p->release_sem) { vkDestroySemaphore(c->device, p->release_sem, NULL); }
    if (p->fence) { vkDestroyFence(c->device, p->fence, NULL); }
    if (p->cmd) { vkFreeCommandBuffers(c->device, c->cmd_pool, 1, &p->cmd); }
    if (p->dpool) { vkDestroyDescriptorPool(c->device, p->dpool, NULL); }
    if (p->sampler) { vkDestroySampler(c->device, p->sampler, NULL); }
    if (p->pipeline) { vkDestroyPipeline(c->device, p->pipeline, NULL); }
    if (p->layout) { vkDestroyPipelineLayout(c->device, p->layout, NULL); }
    if (p->dsl) { vkDestroyDescriptorSetLayout(c->device, p->dsl, NULL); }
    free(p);
}

/* The pass body behind both entry points. `t` is the VK-tensor variant:
 * the destination is its buffer alias and it gets the external hand-off
 * infer_vk_gen gives a tensor -- acquire from / release to
 * VK_QUEUE_FAMILY_EXTERNAL around the dispatch, `released` waited,
 * `ready` signalled -- so Dawn can read it through the vk -> wgpu row. */
static int apply(struct infer_ctx_vk* c,
                 struct hand_frame_pass_vk* p,
                 struct infer_vk_frame_import* im,
                 const struct infer_frame* f,
                 struct hand_affine a,
                 struct hand_norm n,
                 enum hand_border border,
                 uint32_t tw,
                 uint32_t th,
                 VkBuffer dst,
                 uint64_t dst_bytes,
                 VkSemaphore signal,
                 int* release_fd,
                 struct infer_tensor* t) {
    INFER_CHECK(p && im && dst, "hand: vk pass needs an import and a destination buffer");
    VkImageView p0 = infer_vk_frame_plane(im, 0);
    VkImageView p1 = infer_vk_frame_plane(im, 1);
    if (!p1) { p1 = p0; } /* YUYV: one plane, the set must still be complete */
    INFER_CHECK(p0, "hand: vk frame import has no plane 0");
    const int yuyv = f->drm_format == DRM_FORMAT_YUYV;
    struct infer_mem_vk* m = t ? &t->mem.vk : NULL;

    /* the tensor's consumer is done reading it -> a semaphore this submit
     * waits on (a host wait where the fd cannot be imported) */
    VkSemaphore wait_sem = VK_NULL_HANDLE;
    if (m && t->released.sync_fd >= 0) {
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

    /* the previous apply's submit must be done before its descriptor set
     * and command buffer are touched again -- the pass's serialisation.
     * A tensor is submitted on ITS fence, the one its readers wait; a
     * readback may have re-armed it since, so it is waited too. */
    if (hand_frame_vk_wait(c, p) < 0) { return -1; }
    VkFence fence = m ? m->fence : p->fence;
    if (m && !m->first_write) { VK_CHECK(vkWaitForFences(c->device, 1, &fence, VK_TRUE, UINT64_MAX)); }
    VK_CHECK(vkResetFences(c->device, 1, &fence));

    VkDescriptorImageInfo ii[2] = {
        {.sampler = p->sampler, .imageView = p0, .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        {.sampler = p->sampler, .imageView = p1, .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
    };
    VkDescriptorBufferInfo bi = {.buffer = dst, .offset = 0, .range = dst_bytes};
    VkWriteDescriptorSet w[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = p->set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &ii[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = p->set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &ii[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = p->set,
         .dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &bi},
    };
    vkUpdateDescriptorSets(c->device, 3, w, 0, NULL);

    struct pass_params pp;
    memset(&pp, 0, sizeof pp);
    pp.affine[0] = a.m[0];
    pp.affine[1] = a.m[1];
    pp.affine[2] = a.m[3];
    pp.affine[3] = a.m[4];
    pp.extra[0] = a.m[2];
    pp.extra[1] = a.m[5];
    pp.extra[2] = n.lo;
    pp.extra[3] = n.hi - n.lo;
    pp.sizes[0] = tw;
    pp.sizes[1] = th;
    /* the frame's own luma size, not the import's: for YUYV the import is
     * half as wide and the shader converts back through the parity pick */
    pp.sizes[2] = f->width;
    pp.sizes[3] = f->height;
    nv12_yuv_params(f->bt709, f->full_range, pp.coef, pp.range);
    pp.flags[0] = yuyv ? 1u : 0u;
    pp.flags[1] = border == HAND_BORDER_REPLICATE ? 1u : 0u;

    VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(p->cmd, &cbi));
    infer_vk_frame_acquire(c, im, p->cmd);
    /* the tensor: acquire from its external reader (or its first layout) */
    VkImageMemoryBarrier ib;
    VkBufferMemoryBarrier bb;
    if (m) {
        ib = (VkImageMemoryBarrier){
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
        bb = (VkBufferMemoryBarrier){
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .srcQueueFamilyIndex = m->first_write ? VK_QUEUE_FAMILY_IGNORED : VK_QUEUE_FAMILY_EXTERNAL,
            .dstQueueFamilyIndex = m->first_write ? VK_QUEUE_FAMILY_IGNORED : c->queue_family,
            .buffer = m->buffer,
            .size = VK_WHOLE_SIZE,
        };
        vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &bb, 1, &ib);
    }
    vkCmdBindPipeline(p->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
    vkCmdBindDescriptorSets(p->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &p->set, 0,
                            NULL);
    vkCmdPushConstants(p->cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pp, &pp);
    vkCmdDispatch(p->cmd, (tw + 7) / 8, (th + 7) / 8, 1);
    /* make the SSBO writes visible to whoever reads next on this queue
     * (a vk -> cuda copy, the harness's readback staging copy); a host
     * map after the fence is covered by the fence's own host dependency */
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         1, &mb, 0, NULL, 0, NULL);
    if (m) {
        /* release the tensor to its external reader */
        ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ib.dstAccessMask = 0;
        ib.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcQueueFamilyIndex = c->queue_family;
        ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bb.dstAccessMask = 0;
        bb.srcQueueFamilyIndex = c->queue_family;
        bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 1, &bb, 1, &ib);
    }
    VK_CHECK(vkEndCommandBuffer(p->cmd));

    VkSemaphore sig[3];
    uint32_t nsig = 0;
    const int want_release = release_fd != NULL && p->has_release;
    const int want_ready = m != NULL && c->has_sync_fd;
    if (want_release) { sig[nsig++] = p->release_sem; }
    if (signal != VK_NULL_HANDLE) { sig[nsig++] = signal; }
    if (want_ready) { sig[nsig++] = m->done; } /* the tensor's "written" */
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = wait_sem != VK_NULL_HANDLE ? 1 : 0,
        .pWaitSemaphores = &wait_sem,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &p->cmd,
        .signalSemaphoreCount = nsig,
        .pSignalSemaphores = sig,
    };
    VK_CHECK(vkQueueSubmit(c->queue, 1, &si, fence));
    p->fence_armed = 1;
    p->armed_fence = fence;
    if (m) { m->first_write = 0; }

    if (release_fd) {
        *release_fd = -1;
        if (want_release) {
            VkSemaphoreGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                                           .semaphore = p->release_sem,
                                           .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
            if (c->get_semaphore_fd(c->device, &gfi, release_fd) != VK_SUCCESS) {
                *release_fd = -1;
            }
        }
    }
    if (m) {
        /* the tensor's ready token, as infer_vk_gen hands it out */
        int fd = -1;
        if (want_ready) {
            VkSemaphoreGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                                           .semaphore = m->done,
                                           .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
            if (c->get_semaphore_fd(c->device, &gfi, &fd) != VK_SUCCESS) { fd = -1; }
        } else {
            VK_CHECK(vkWaitForFences(c->device, 1, &fence, VK_TRUE, UINT64_MAX));
        }
        infer_sync_set(&t->ready, fd);
    }
    /* submitted, not complete: the fence (or `signal`) is the token */
    return 0;
}

int hand_frame_to_tensor_vk(struct infer_ctx_vk* c,
                            struct hand_frame_pass_vk* p,
                            struct infer_vk_frame_import* im,
                            const struct infer_frame* f,
                            struct hand_affine a,
                            struct hand_norm n,
                            enum hand_border border,
                            uint32_t tw,
                            uint32_t th,
                            VkBuffer dst,
                            uint64_t dst_bytes,
                            VkSemaphore signal,
                            int* release_fd) {
    return apply(c, p, im, f, a, n, border, tw, th, dst, dst_bytes, signal, release_fd, NULL);
}

int hand_frame_to_tensor_vk_tensor(struct infer_ctx_vk* c,
                                   struct hand_frame_pass_vk* p,
                                   struct infer_vk_frame_import* im,
                                   const struct infer_frame* f,
                                   struct hand_affine a,
                                   struct hand_norm n,
                                   enum hand_border border,
                                   struct infer_tensor* t,
                                   int* release_fd) {
    INFER_CHECK(t->domain == INFER_DOMAIN_VK && t->mem.vk.buffer,
                "hand: the vk pass writes a VK tensor through its buffer alias");
    INFER_CHECK(t->desc.ndim == 4 && t->desc.dims[3] == 3, "hand: tensor is not [1,H,W,3]");
    return apply(c, p, im, f, a, n, border, (uint32_t)t->desc.dims[2], (uint32_t)t->desc.dims[1],
                 t->mem.vk.buffer, infer_desc_bytes_packed(&t->desc), VK_NULL_HANDLE, release_fd,
                 t);
}
