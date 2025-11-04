glm::mat4 M = /* model */; glm::mat4 V = /* view */;
glm::mat4 P = /* projection */;
// compose once on CPU to reduce per-vertex ALU work
glm::mat4 PVM = P * V * M;                     // column-major, column vectors
glUniformMatrix4fv(location_uMVP, 1, GL_FALSE, &PVM[0][0]); // upload as uniform