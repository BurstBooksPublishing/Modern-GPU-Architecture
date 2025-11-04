vkBindVertexBuffer(cmdBuffer, 0, vertexBuffer); // bind vertex attributes
vkBindIndexBuffer(cmdBuffer, indexBuffer, VK_INDEX_TYPE_UINT16); // bind indices
vkCmdDrawIndexed(cmdBuffer, indexCount, 1, 0, 0, 0); // issue indexed draw