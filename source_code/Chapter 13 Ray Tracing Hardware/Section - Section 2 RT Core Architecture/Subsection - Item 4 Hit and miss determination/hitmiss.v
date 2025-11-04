module hit_miss_unit #(
  parameter FP_WIDTH = 32
)(
  input  wire                  clk,
  input  wire                  rst_n,
  // input candidate from triangle unit
  input  wire                  in_valid,
  output wire                  in_ready,
  input  wire [FP_WIDTH-1:0]   in_t,
  input  wire [31:0]           in_prim_mask,
  input  wire                  in_tri_valid, // barycentrics already checked
  // ray state
  input  wire [FP_WIDTH-1:0]   ray_tmin,
  input  wire [FP_WIDTH-1:0]   ray_tmax,
  input  wire [31:0]           ray_mask,
  input  wire                  mode_any_hit, // 1 = any-hit (shadow), 0 = closest
  // outputs
  output reg                   out_hit_valid,
  output reg  [FP_WIDTH-1:0]   out_hit_t,
  output reg  [31:0]           out_hit_prim_mask,
  input  wire                  out_ready
);

assign in_ready = 1'b1; // simple pipeline stage: always ready

wire mask_pass = |(ray_mask & in_prim_mask); // bitwise AND non-zero
wire t_in_range;
assign t_in_range = (in_t >= ray_tmin) && (in_t <= ray_tmax);

reg [FP_WIDTH-1:0] t_best;
reg               found_any;

// combinational hit detection
wire candidate_hit = in_valid && in_tri_valid && mask_pass && t_in_range;

always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    out_hit_valid    <= 1'b0;
    out_hit_t        <= {FP_WIDTH{1'b0}};
    out_hit_prim_mask<= 32'b0;
    t_best           <= {FP_WIDTH{1'b1}}; // init to max value
    found_any        <= 1'b0;
  end else begin
    out_hit_valid <= 1'b0;
    if (candidate_hit) begin
      if (mode_any_hit) begin
        // any-hit: output immediately and mark found
        out_hit_valid    <= 1'b1;
        out_hit_t        <= in_t;
        out_hit_prim_mask<= in_prim_mask;
        found_any        <= 1'b1;
      end else begin
        // closest-hit: compare and update
        if (in_t < t_best) begin
          t_best           <= in_t;
          out_hit_valid    <= out_ready ? 1'b1 : 1'b0; // flow control
          out_hit_t        <= in_t;
          out_hit_prim_mask<= in_prim_mask;
        end
      end
    end
    // optional: clear found_any when traversal restarts externally
  end
end

endmodule