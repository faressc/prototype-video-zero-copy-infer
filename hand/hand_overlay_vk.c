/* hand_overlay_vk.c -- see hand_overlay_vk.h. Modelled on
 * scene_cam_vk.c's create_pipeline, minus everything the overlay does
 * not need: no descriptor set, no sampler, no depth. */
#include <string.h>

#include "hand_overlay_vk.h"
#include "infer_util.h" /* VK_CHECK */

#ifndef SHADER_DIR
#define SHADER_DIR "./shaders"
#endif

enum { OVERLAY_VERTS = HAND_OVERLAY_MAX_VERTS };

struct overlay_push {
    float rect[4]; /* xy = scale, zw = offset */
};

void hand_overlay_vk_init(struct hand_overlay_vk* g, struct vk_presenter* p) {
    memset(g, 0, sizeof *g);
    g->capacity = OVERLAY_VERTS;

    VkShaderModule vs = vk_present_load_shader(p, SHADER_DIR "/overlay.vert.spv");
    VkShaderModule fs = vk_present_load_shader(p, SHADER_DIR "/overlay.frag.spv");

    VkPushConstantRange pc = {.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                              .offset = 0,
                              .size = sizeof(struct overlay_push)};
    /* setLayoutCount 0: a vertex buffer and 16 bytes is the whole input */
    VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                      .setLayoutCount = 0,
                                      .pushConstantRangeCount = 1,
                                      .pPushConstantRanges = &pc};
    VK_CHECK(vkCreatePipelineLayout(p->device, &lci, NULL, &g->layout));

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
    VkVertexInputBindingDescription bind = {.binding = 0,
                                           .stride = sizeof(struct hand_vert),
                                           .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
    /* R8G8B8A8_UNORM reads the four bytes in memory order as r,g,b,a --
     * which is exactly what hand_rgba packs, so the palette constants
     * mean the same thing here as in the GLES path */
    VkVertexInputAttributeDescription attrs[2] = {
        {.location = 0,
         .binding = 0,
         .format = VK_FORMAT_R32G32_SFLOAT,
         .offset = offsetof(struct hand_vert, x)},
        {.location = 1,
         .binding = 0,
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .offset = offsetof(struct hand_vert, rgba)},
    };
    VkPipelineVertexInputStateCreateInfo vin = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bind,
        .vertexAttributeDescriptionCount = 2,
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
        /* the strokes' winding flips as a stroke turns, so cull nothing */
        .cullMode = VK_CULL_MODE_NONE,
        .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState blend_att = {
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
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
    /* the camera scene's scratch images share the swapchain's format, so
     * one pipeline serves the window pass and the scratch pass alike */
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
        .layout = g->layout,
    };
    VK_CHECK(vkCreateGraphicsPipelines(p->device, VK_NULL_HANDLE, 1, &pci, NULL, &g->pipeline));
    vkDestroyShaderModule(p->device, vs, NULL);
    vkDestroyShaderModule(p->device, fs, NULL);

    /* one host-visible vertex buffer per lane, mapped for the process's
     * life: the geometry is rewritten every frame and never read back */
    const VkDeviceSize size = (VkDeviceSize)g->capacity * sizeof(struct hand_vert);
    for (int i = 0; i < VK_PRESENT_LANES; i++) {
        VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .size = size,
                                  .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                  .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VK_CHECK(vkCreateBuffer(p->device, &bci, NULL, &g->vbo[i]));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(p->device, g->vbo[i], &req);
        VkMemoryAllocateInfo mai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = req.size,
            .memoryTypeIndex = vk_present_find_memory_type(
                p,
                req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        };
        VK_CHECK(vkAllocateMemory(p->device, &mai, NULL, &g->mem[i]));
        VK_CHECK(vkBindBufferMemory(p->device, g->vbo[i], g->mem[i], 0));
        VK_CHECK(vkMapMemory(p->device, g->mem[i], 0, size, 0, &g->mapped[i]));
    }
}

void hand_overlay_vk_fini(struct hand_overlay_vk* g, struct vk_presenter* p) {
    for (int i = 0; i < VK_PRESENT_LANES; i++) {
        if (g->mem[i]) {
            vkUnmapMemory(p->device, g->mem[i]);
            vkFreeMemory(p->device, g->mem[i], NULL);
        }
        if (g->vbo[i]) { vkDestroyBuffer(p->device, g->vbo[i], NULL); }
    }
    if (g->pipeline) { vkDestroyPipeline(p->device, g->pipeline, NULL); }
    if (g->layout) { vkDestroyPipelineLayout(p->device, g->layout, NULL); }
    memset(g, 0, sizeof *g);
}

void hand_overlay_vk_draw(struct hand_overlay_vk* g,
                          const struct vk_frame* fr,
                          const struct hand_vert* v,
                          int n,
                          float sx,
                          float sy,
                          float ox,
                          float oy) {
    if (n <= 0 || !g->pipeline) { return; }
    if (n > g->capacity) { n = g->capacity; }
    memcpy(g->mapped[fr->lane], v, (size_t)n * sizeof *v);

    struct overlay_push push = {{sx, sy, ox, oy}};
    vkCmdBindPipeline(fr->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g->pipeline);
    vkCmdPushConstants(fr->cmd, g->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof push, &push);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fr->cmd, 0, 1, &g->vbo[fr->lane], &offset);
    vkCmdDraw(fr->cmd, (uint32_t)n, 1, 0, 0);
}
