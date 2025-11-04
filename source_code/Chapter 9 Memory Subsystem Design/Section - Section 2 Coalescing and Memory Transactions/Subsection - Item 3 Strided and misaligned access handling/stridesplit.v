module stride_tx_gen #(
  parameter ADDR_W = 64,
  parameter COUNT_W = 16,
  parameter CL_BYTES = 128
) (
  input  wire                   clk,
  input  wire                   rst,
  input  wire                   start,                   // start request
  input  wire [ADDR_W-1:0]      base_addr,               // A0
  input  wire [ADDR_W-1:0]      stride,                  // s
  input  wire [15:0]            elem_bytes,              // b
  input  wire [COUNT_W-1:0]     elem_count,              // N
  input  wire                   mode_span,               // 1 -> span, 0 -> per-line
  input  wire                   ready_in,                // consumer ready
  output reg                    busy,
  output reg                    trans_valid,
  output reg [ADDR_W-1:0]       trans_addr,              // aligned start
  output reg [15:0]             trans_bytes,             // normally CL_BYTES
  output reg                    trans_last
);
  // local calculations
  localparam integer CLW = CL_BYTES;
  reg [ADDR_W-1:0] min_addr, max_addr, aligned_start, aligned_end;
  reg [COUNT_W-1:0] idx;
  reg [ADDR_W-1:0] cur_addr;
  // compute endpoints when start asserted
  always @(posedge clk) begin
    if (rst) begin
      busy <= 0; trans_valid <= 0; trans_last <= 0;
    end else begin
      if (start && !busy) begin
        busy <= 1;
        min_addr <= base_addr;
        max_addr <= base_addr + (elem_count-1) * stride + (elem_bytes - 1);
        aligned_start <= (base_addr / CLW) * CLW;
        aligned_end   <= (max_addr / CLW) * CLW;
        idx <= 0;
        cur_addr <= (base_addr / CLW) * CLW;
        trans_valid <= 0;
      end else if (busy) begin
        if (mode_span) begin
          // issue contiguous aligned lines from aligned_start..aligned_end
          if (!trans_valid && ready_in) begin
            trans_addr  <= cur_addr;
            trans_bytes <= CL_BYTES;
            trans_last  <= (cur_addr == aligned_end);
            trans_valid <= 1;
          end else if (trans_valid && ready_in) begin
            trans_valid <= 0;
            if (cur_addr == aligned_end) begin
              busy <= 0; // done
            end else begin
              cur_addr <= cur_addr + CLW;
            end
          end
        end else begin
          // per-element line issuance (may duplicate lines if stride small;
          // a small hardware dedupe can be added externally)
          if (!trans_valid && ready_in) begin
            trans_addr  <= ((base_addr + idx*stride) / CLW) * CLW;
            trans_bytes <= CL_BYTES;
            trans_last  <= (idx == elem_count-1);
            trans_valid <= 1;
          end else if (trans_valid && ready_in) begin
            trans_valid <= 0;
            if (idx == elem_count-1) busy <= 0; else idx <= idx + 1;
          end
        end
      end
    end
  end
endmodule