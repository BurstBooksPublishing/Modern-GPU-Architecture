module attr_fetch #(
  parameter ADDR_W = 32,
  parameter DATA_W = 128  // 4x32 FP32 per return
)(
  input  wire                  clk,
  input  wire                  rst_n,
  // request: vertex index + descriptor valid
  input  wire                  req_valid,
  input  wire [31:0]           req_index,  // vertex index
  input  wire [ADDR_W-1:0]     desc_base,  // attribute base address
  input  wire [15:0]           desc_stride,// per-vertex stride (bytes)
  input  wire [15:0]           desc_offset,// attribute offset within vertex
  output reg                   req_ready,
  // memory master interface (simple)
  output reg                   mem_req_valid,
  output reg [ADDR_W-1:0]      mem_req_addr,
  input  wire                  mem_req_ready,
  input  wire                  mem_rsp_valid,
  input  wire [DATA_W-1:0]     mem_rsp_data,
  // output attribute
  output reg                   out_valid,
  output reg [DATA_W-1:0]      out_attr,
  input  wire                  out_ready
);
  // Stage 0: accept request
  reg [31:0] idx_r;
  reg [ADDR_W-1:0] base_r;
  reg [15:0] stride_r, off_r;
  reg busy;
  always @(posedge clk) begin
    if (!rst_n) begin
      req_ready <= 1'b1; busy <= 1'b0;
      mem_req_valid <= 1'b0; out_valid <= 1'b0;
    end else begin
      if (req_valid && req_ready) begin
        idx_r <= req_index; base_r <= desc_base;
        stride_r <= desc_stride; off_r <= desc_offset;
        req_ready <= 1'b0; busy <= 1'b1;
      end
      // Stage 1: compute address and issue mem req
      if (busy && !mem_req_valid) begin
        mem_req_addr <= base_r + idx_r*stride_r + off_r; // Eq. (\ref{eq:addr})
        mem_req_valid <= 1'b1;
      end
      if (mem_req_valid && mem_req_ready) mem_req_valid <= 1'b0;
      // Stage 2: respond on memory return
      if (mem_rsp_valid) begin
        out_attr <= mem_rsp_data; out_valid <= 1'b1;
      end
      if (out_valid && out_ready) begin
        out_valid <= 1'b0; busy <= 1'b0; req_ready <= 1'b1;
      end
    end
  end
endmodule