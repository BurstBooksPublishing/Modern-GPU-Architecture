module format_adapter #(
  parameter DATA_W = 32 // output bus width
) (
  input  clk,
  input  rstn,
  // Y-plane stream (byte lanes valid)
  input  y_valid,
  input  [7:0] y_data,
  output y_ready,
  // UV-plane stream (interleaved U then V bytes)
  input  uv_valid,
  input  [7:0] uv_data,
  output uv_ready,
  // Config: 0 = NV12 8-bit, 1 = P010 10-bit (packed into 16-bit lanes)
  input  mode_10bit,
  // Output stream (DATA_W aligned)
  output reg out_valid,
  output reg [DATA_W-1:0] out_data,
  input  out_ready
);
  // Simple FIFOs implied by ready signals; here flow-through pass-through logic.
  assign y_ready  = out_ready; // backpressure prop
  assign uv_ready = out_ready;

  // Pack two pixels per DATA_W when 8-bit; in 10-bit mode pack into 16-bit lanes.
  reg [15:0] pack_buf;
  reg       pack_half;
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      out_valid <= 0;
      out_data  <= 0;
      pack_buf  <= 0;
      pack_half <= 0;
    end else begin
      out_valid <= 0;
      if (y_valid && uv_valid && out_ready) begin
        if (!mode_10bit) begin
          // NV12: output 4 bytes per 32-bit word: Y0 Y1 U V
          out_data <= {y_data, y_data, uv_data, uv_data}; // simple repeated example
          out_valid <= 1;
        end else begin
          // P010-like: pack 10-bit into upper bits of 16-bit lanes
          // Here we zero-extend 8->10 bits for simplicity.
          pack_buf <= {6'b0, y_data}; // 16-bit lane containing 10-bit sample
          out_data <= {pack_buf, {6'b0, uv_data}}; // two 16-bit lanes form 32-bit word
          out_valid <= 1;
        end
      end
    end
  end
endmodule