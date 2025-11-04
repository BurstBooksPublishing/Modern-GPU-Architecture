module link_aggregator #(
  parameter FLIT_W = 128
)(
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire [FLIT_W-1:0]    in_flit,
  input  wire                 in_valid,
  output wire                 in_ready,
  // link 0
  output reg  [FLIT_W-1:0]    link0_flit,
  output reg                  link0_valid,
  input  wire                 link0_credit, // credit returned when consumed
  // link 1
  output reg  [FLIT_W-1:0]    link1_flit,
  output reg                  link1_valid,
  input  wire                 link1_credit
);

  // simple credit counters (1-bit per link for example)
  reg link0_busy, link1_busy;
  reg toggle; // round-robin striping bit

  assign in_ready = ~((toggle & link1_busy) | (~toggle & link0_busy));

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      link0_flit <= {FLIT_W{1'b0}};
      link1_flit <= {FLIT_W{1'b0}};
      link0_valid <= 1'b0;
      link1_valid <= 1'b0;
      link0_busy <= 1'b0;
      link1_busy <= 1'b0;
      toggle <= 1'b0;
    end else begin
      // accept and stripe when input valid and target link free
      if (in_valid && in_ready) begin
        if (~toggle && ~link0_busy) begin
          link0_flit <= in_flit;
          link0_valid <= 1'b1;
          link0_busy <= 1'b1;
          toggle <= ~toggle;
        end else if (toggle && ~link1_busy) begin
          link1_flit <= in_flit;
          link1_valid <= 1'b1;
          link1_busy <= 1'b1;
          toggle <= ~toggle;
        end
      end
      // release when credits observed
      if (link0_credit) begin link0_valid <= 1'b0; link0_busy <= 1'b0; end
      if (link1_credit) begin link1_valid <= 1'b0; link1_busy <= 1'b0; end
    end
  end
endmodule