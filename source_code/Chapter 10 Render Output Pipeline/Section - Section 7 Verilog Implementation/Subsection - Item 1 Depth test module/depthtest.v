module depth_test #(
  parameter DEPTH_BITS = 24
)(
  input  wire                     clk,
  input  wire                     rst_n,
  // input fragment channel
  input  wire                     in_valid,
  output reg                      in_ready,
  input  wire [DEPTH_BITS-1:0]    frag_z,
  // memory readback (from depth buffer)
  input  wire [DEPTH_BITS-1:0]    z_mem,
  // configuration
  input  wire [2:0]               cmp_mode,          // encoded compare
  input  wire                     depth_write_en,
  input  wire                     depth_bounds_en,
  input  wire [DEPTH_BITS-1:0]    depth_bounds_min,
  input  wire [DEPTH_BITS-1:0]    depth_bounds_max,
  // output: pass + optional write data
  output reg                      out_valid,
  input  wire                     out_ready,
  output reg                      out_pass,
  output reg [DEPTH_BITS-1:0]     out_write_z
);

  // compare mode encoding
  localparam CMP_LE    = 3'd0;
  localparam CMP_LEQ   = 3'd1;
  localparam CMP_GE    = 3'd2;
  localparam CMP_GEQ   = 3'd3;
  localparam CMP_EQ    = 3'd4;
  localparam CMP_NEQ   = 3'd5;
  localparam CMP_ALWAYS= 3'd6;

  // combinational compare and bounds
  wire cmp_le   = (frag_z < z_mem);
  wire cmp_leq  = (frag_z <= z_mem);
  wire cmp_ge   = (frag_z > z_mem);
  wire cmp_geq  = (frag_z >= z_mem);
  wire cmp_eq   = (frag_z == z_mem);
  wire cmp_neq  = (frag_z != z_mem);
  wire bounds_ok = (!depth_bounds_en) ||
                   ((frag_z >= depth_bounds_min) && (frag_z <= depth_bounds_max));

  reg pass_comb;
  always @(*) begin
    case (cmp_mode)
      CMP_LE:    pass_comb = cmp_le;
      CMP_LEQ:   pass_comb = cmp_leq;
      CMP_GE:    pass_comb = cmp_ge;
      CMP_GEQ:   pass_comb = cmp_geq;
      CMP_EQ:    pass_comb = cmp_eq;
      CMP_NEQ:   pass_comb = cmp_neq;
      CMP_ALWAYS:pass_comb = 1'b1;
      default:   pass_comb = 1'b0;
    endcase
    pass_comb = pass_comb & bounds_ok;
  end

  // pipeline registers + valid-ready protocol
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      in_ready  <= 1'b1;
      out_valid <= 1'b0;
      out_pass  <= 1'b0;
      out_write_z <= {DEPTH_BITS{1'b0}};
    end else begin
      // accept when downstream not backpressured
      in_ready <= out_ready | ~out_valid;
      if (in_valid && in_ready) begin
        out_valid <= 1'b1;
        out_pass  <= pass_comb;
        out_write_z <= (depth_write_en && pass_comb) ? frag_z : z_mem;
      end else if (out_valid && out_ready) begin
        out_valid <= 1'b0;
      end
    end
  end

endmodule