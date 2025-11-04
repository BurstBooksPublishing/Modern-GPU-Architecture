module depth_bounds_unit #(
  parameter WIDTH = 24
)(
  input  wire [WIDTH-1:0] tile_z_min, // tile min Z metadata
  input  wire [WIDTH-1:0] tile_z_max, // tile max Z metadata
  input  wire [WIDTH-1:0] bound_min,  // draw-level min bound
  input  wire [WIDTH-1:0] bound_max,  // draw-level max bound
  input  wire [WIDTH-1:0] frag_z,     // per-fragment Z
  output wire              tile_reject,
  output wire              tile_accept,
  output wire              frag_pass      // per-fragment bounds pass
);
  // full-tile decisions (Equation \ref{eq:bounds})
  assign tile_reject = (tile_z_max < bound_min) || (tile_z_min > bound_max);
  assign tile_accept = (tile_z_min >= bound_min) && (tile_z_max <= bound_max);
  // per-fragment compare executed when tile inconclusive
  assign frag_pass = (frag_z >= bound_min) && (frag_z <= bound_max);
endmodule