/* scene_cube_vk.c -- a lit, spinning cube on the Vulkan presenter.
 *
 * What is new versus the gradient, each the explicit form of something
 * the GLES scene gets in one call:
 *
 *   VkBuffer + VkDeviceMemory       glBufferData -- create, query
 *                                   requirements, pick a memory type,
 *                                   allocate, bind, map, copy
 *   VkPipelineVertexInputState      glVertexAttribPointer, frozen into
 *                                   the pipeline: the form that was
 *                                   empty for the gradient
 *   VkPipelineDepthStencilState     glEnable(GL_DEPTH_TEST)
 *   cullMode / frontFace            glCullFace / glFrontFace
 *   two mat4 push constants         glUniformMatrix4fv -- 128 bytes, the
 *                                   guaranteed minimum, fully spent
 *   depth attachment in             the presenter's depth image, plugged
 *   VkRenderingInfo                 in next to the color one
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cube_mesh.h"
#include "mat4.h"
#include "vk_present.h"

#ifndef SHADER_DIR
#define SHADER_DIR "./shaders"
#endif

struct cube_push { /* must match the push_constant block in cube.vert */
    mat4 mvp;
    mat4 model;
};

struct cube {
    VkPipelineLayout layout;
    VkPipeline pipeline;
    VkBuffer vbo, ibo;
    VkDeviceMemory vbo_mem, ibo_mem;
};

/* Memory in Vulkan is a separate object from the resource using it.
 * Host-visible + coherent: the CPU maps it and writes directly, no
 * staging copy -- on unified-memory GPUs (AGX) this IS the fast path;
 * on discrete GPUs a real engine would upload to DEVICE_LOCAL memory
 * through a staging buffer + vkCmdCopyBuffer. */
static void create_host_buffer(struct vk_presenter* p,
                               VkBufferUsageFlags usage,
                               const void* data,
                               VkDeviceSize size,
                               VkBuffer* buf,
                               VkDeviceMemory* mem) {
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(p->device, &bci, NULL, buf));

    VkMemoryRequirements req; /* size incl. alignment padding + allowed types */
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
    vkUnmapMemory(p->device, *mem); /* coherent: no flush needed */
}

static void create_pipeline(struct vk_presenter* p, struct cube* c) {
    VkShaderModule vs = vk_present_load_shader(p, SHADER_DIR "/cube.vert.spv");
    VkShaderModule fs = vk_present_load_shader(p, SHADER_DIR "/cube.frag.spv");

    VkPushConstantRange pc_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(struct cube_push),
    };
    VkPipelineLayoutCreateInfo lci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc_range,
    };
    VK_CHECK(vkCreatePipelineLayout(p->device, &lci, NULL, &c->layout));

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

    /* The vertex fetch form, filled in: one buffer binding, three
     * attributes reading it at byte offsets -- glVertexAttribPointer x3,
     * declared once instead of per draw. */
    VkVertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(struct cube_vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    VkVertexInputAttributeDescription attrs[3] = {
        {.location = 0,
         .binding = 0,
         .format = VK_FORMAT_R32G32B32_SFLOAT,
         .offset = offsetof(struct cube_vertex, pos)},
        {.location = 1,
         .binding = 0,
         .format = VK_FORMAT_R32G32B32_SFLOAT,
         .offset = offsetof(struct cube_vertex, normal)},
        {.location = 2,
         .binding = 0,
         .format = VK_FORMAT_R32G32B32_SFLOAT,
         .offset = offsetof(struct cube_vertex, color)},
    };
    VkPipelineVertexInputStateCreateInfo vin = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 3,
        .pVertexAttributeDescriptions = attrs,
    };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1};
    /* Culling on. The mesh is CCW from outside; the projection flips y
     * for Vulkan's y-down clip space, so on screen it is drawn CCW --
     * and Vulkan judges winding as seen on screen. */
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    /* The depth socket, in use: test + write, nearer wins. */
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
        .layout = c->layout,
    };
    VK_CHECK(vkCreateGraphicsPipelines(p->device, VK_NULL_HANDLE, 1, &pci, NULL, &c->pipeline));

    vkDestroyShaderModule(p->device, vs, NULL);
    vkDestroyShaderModule(p->device, fs, NULL);
}

static void cube_init(struct vk_presenter* p, void* user) {
    struct cube* c = user;
    create_host_buffer(p,
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                       cube_vertices,
                       sizeof(cube_vertices),
                       &c->vbo,
                       &c->vbo_mem);
    create_host_buffer(p,
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                       cube_indices,
                       sizeof(cube_indices),
                       &c->ibo,
                       &c->ibo_mem);
    create_pipeline(p, c);
}

static void cube_record(struct vk_presenter* p, const struct vk_frame* fr, void* user) {
    struct cube* c = user;
    VkCommandBuffer cmd = fr->cmd;

    /* Two canvases this time. Color: clear (the cube does not cover
     * every pixel, so DONT_CARE would leave garbage). Depth: clear to
     * far, and DONT_CARE on store -- nobody reads it after the pass. */
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
    vkCmdBeginRendering(cmd, &ri);

    VkViewport viewport = {0, 0, (float)fr->extent.width, (float)fr->extent.height, 0.0f, 1.0f};
    VkRect2D scissor = {{0, 0}, fr->extent};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, c->pipeline);

    /* The buffers the vertex-input form described, bound for this draw. */
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &c->vbo, &offset);
    vkCmdBindIndexBuffer(cmd, c->ibo, 0, VK_INDEX_TYPE_UINT16);

    struct cube_push push;
    push.model = cube_model_matrix(&p->app);
    push.mvp =
        mat4_mul(cube_view_proj((int)fr->extent.width, (int)fr->extent.height, 1, 1, p->app.zoom),
                 push.model);
    vkCmdPushConstants(cmd, c->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);

    /* gl_VertexIndex now comes from the index buffer: 36 lookups into
     * 24 vertices. */
    vkCmdDrawIndexed(cmd, CUBE_INDEX_COUNT, 1, 0, 0, 0);

    vkCmdEndRendering(cmd);
}

static void cube_fini(struct vk_presenter* p, void* user) {
    struct cube* c = user;
    vkDestroyPipeline(p->device, c->pipeline, NULL);
    vkDestroyPipelineLayout(p->device, c->layout, NULL);
    vkDestroyBuffer(p->device, c->vbo, NULL);
    vkFreeMemory(p->device, c->vbo_mem, NULL);
    vkDestroyBuffer(p->device, c->ibo, NULL);
    vkFreeMemory(p->device, c->ibo_mem, NULL);
}

static const struct vk_scene cube_scene = {
    .init = cube_init,
    .resize = NULL,
    .record = cube_record,
    .fini = cube_fini,
};

int main(void) {
    struct cube c = {0};
    struct vk_presenter p;

    if (vk_present_init(&p,
                        VK_PRESENT_DEPTH,
                        &cube_scene,
                        &c,
                        "cube (Vulkan)",
                        "hello-wayland-cube-vulkan") < 0) {
        return 1;
    }
    vk_present_run(&p);
    vk_present_fini(&p);
    return 0;
}
