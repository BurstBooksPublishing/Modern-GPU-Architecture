module interp_pipe #(
  parameter WIDTH = 32,
  parameter FRAC  = 16
) (
  input  wire                  clk,
  input  wire                  rst_n,
  // input handshake
  input  wire                  in_valid,
  output wire                  in_ready,
  input  wire [WIDTH-1:0]      init_v,     // Qx.FRAC: starting v/w
  input  wire [WIDTH-1:0]      init_q,     // Qx.FRAC: starting 1/w
  input  wire [WIDTH-1:0]      dv_dx,      // gradient per x
  input  wire [WIDTH-1:0]      dq_dx,
  // output handshake
  output reg                   out_valid,
  input  wire                  out_ready,
  output reg  [WIDTH-1:0]      interp_v    // final interpolated v (Qx.FRAC)
);
  // Stage 0: latch inputs
  reg [WIDTH-1:0] s0_v, s0_q, s0_dv, s0_dq;
  reg             s0_valid;
  assign in_ready = !s0_valid;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s0_valid <= 1'b0;
    end else if (in_valid && in_ready) begin
      s0_v    <= init_v; s0_q <= init_q;
      s0_dv   <= dv_dx;  s0_dq <= dq_dx;
      s0_valid<= 1'b1;
    end else if (s0_valid && !in_ready) begin
      // stay
    end else if (!in_valid) begin
      s0_valid <= s0_valid; // hold
    end
  end

  // Stage 1: incremental update (per-pixel step)
  reg [WIDTH-1:0] s1_v, s1_q;
  reg             s1_valid;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s1_valid<=1'b0;
    end else begin
      if (s0_valid) begin
        s1_v    <= s0_v + s0_dv;   // next pixel value
        s1_q    <= s0_q + s0_dq;
        s1_valid<=1'b1;
        s0_valid<=1'b0;
      end else if (s1_valid && out_ready && !out_valid) begin
        s1_valid<=1'b0;
      end
    end
  end

  // Stage 2: final divide (perspective correction)
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
      interp_v  <= {WIDTH{1'b0}};
    end else begin
      if (s1_valid && !out_valid) begin
        // combinational divide (synthesizable / operator)
        interp_v  <= s1_v / s1_q;
        out_valid <= 1'b1;
      end else if (out_valid && out_ready) begin
        out_valid <= 1'b0;
      end
    end
  end
endmodule