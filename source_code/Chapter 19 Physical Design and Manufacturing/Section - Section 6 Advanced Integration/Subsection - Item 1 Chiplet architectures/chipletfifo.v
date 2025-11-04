module chiplet_link_endpoint #(
  parameter WIDTH = 64,
  parameter DEPTH = 16,
  parameter PTR_W = $clog2(DEPTH)
)(
  input  wire                 clk,
  input  wire                 rst,
  // Ingress (from remote chiplet)
  input  wire [WIDTH-1:0]     in_data,
  input  wire                 in_valid,
  output wire                 in_ready, // credit-driven
  // Egress (to local consumer)
  output reg  [WIDTH-1:0]     out_data,
  output reg                  out_valid,
  input  wire                 out_ready
);
  // circular buffer storage
  reg [WIDTH-1:0] ram [0:DEPTH-1];
  reg [PTR_W-1:0] wr_ptr, rd_ptr;
  reg [PTR_W:0]   count;

  assign in_ready = (count < DEPTH); // accept when space exists

  // write side (ingress)
  always @(posedge clk) begin
    if (rst) begin
      wr_ptr <= 0; count <= 0;
    end else if (in_valid && in_ready) begin
      ram[wr_ptr] <= in_data; wr_ptr <= wr_ptr + 1; count <= count + 1;
    end
  end

  // read side (egress)
  always @(posedge clk) begin
    if (rst) begin
      rd_ptr <= 0; out_valid <= 0; out_data <= 0;
    end else begin
      if (!out_valid && (count > 0)) begin
        out_data <= ram[rd_ptr];
        out_valid <= 1;
        rd_ptr <= rd_ptr + 1;
        count <= count - 1;
      end else if (out_valid && out_ready) begin
        out_valid <= 0;
      end
    end
  end
endmodule