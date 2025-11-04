/* compactIndices: parallel prefix-sum compaction helper */
for (int depth=0; depth>>(trace_queue, M); // RT core calls
  launch_shade_kernel<<>>(trace_queue, M); // shading, may push to scatter_queue
  // scatter_queue becomes next depth's active_indices
  swap(active_indices, scatter_queue);
  if (active_count==0) break; // early exit when rays terminated
}