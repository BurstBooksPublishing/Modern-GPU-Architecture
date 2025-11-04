module early_z_unit #(
  parameter DEPTH_BITS = 24
)(
  input  wire                  clk,
  input  wire                  rst_n,
  input  wire                  valid_in,        // incoming fragment valid
  input  wire [DEPTH_BITS-1:0] frag_depth,      // interpolated depth
  input  wire [DEPTH_BITS-1:0] stored_depth,    // current Z from ROP/L2
  input  wire                  depth_write_en,  // permit depth updates
  output reg                   ready_in,        // backpressure
  output reg                   pass,            // fragment may proceed
  output reg  [DEPTH_BITS-1:0] depth_out        // depth to write if enabled
);
  // simple pipeline stage: accept when ready; one-cycle compare
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ready_in <= 1'b1;
      pass     <= 1'b0;
      depth_out<= {DEPTH_BITS{1'b0}};
    end else begin
      if (valid_in && ready_in) begin
        // depth test: less-than (typical depth func)
        if (frag_depth < stored_depth) begin
          pass <= 1'b1;
          if (depth_write_en) depth_out <= frag_depth; // schedule write
        end else begin
          pass <= 1'b0;
        end
        ready_in <= 1'b1; // single-cycle accept for simplicity
      end
    end
  end
endmodule