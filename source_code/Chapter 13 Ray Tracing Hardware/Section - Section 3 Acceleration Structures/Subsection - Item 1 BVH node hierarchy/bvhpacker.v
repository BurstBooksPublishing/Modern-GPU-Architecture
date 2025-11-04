module bvh_node_packer(
  input  wire        is_leaf,                 // 1=leaf, 0=interior
  input  wire [1:0]  axis,                    // split axis for interior
  input  wire [14:0] child0_off, child1_off,  // relative offsets for interior
  input  wire [19:0] prim_index,              // base primitive index for leaf
  input  wire [11:0] prim_count,              // primitive count for leaf
  input  wire [15:0] min_x, min_y, min_z,     // quantized bbox min
  input  wire [15:0] max_x, max_y, max_z,     // quantized bbox max
  output wire [127:0] node_word               // packed node output
);
  wire [31:0] union_field;
  // interior: [31]=0 reserved, [30:15]=child0, [14:0]=child1 with axis in top bits
  // leaf:   [31:12]=prim_index, [11:0]=prim_count
  assign union_field = is_leaf ?
    {prim_index, prim_count} :
    {1'b0, axis, child0_off, child1_off}; // packs axis and two 15-bit offsets
  assign node_word = {max_z, max_y, max_x, min_z, min_y, min_x, union_field};
endmodule