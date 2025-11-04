module bitstream_parser (
  input  wire        clk,
  input  wire        rst_n,
  input  wire [7:0]  in_byte,    // input byte stream
  input  wire        in_valid,
  output reg         in_ready,
  // read interface
  input  wire [4:0]  req_bits,   // request bits (1..32)
  input  wire        req_valid,
  output reg  [31:0] req_data,
  output reg         req_ready,
  // exp-golomb request
  input  wire        req_ue_valid,
  output reg  [15:0] ue_value,
  output reg         ue_ready,
  // huffman output (max 16-bit code)
  output reg  [15:0] huff_sym,
  output reg         huff_valid
);
  // internal 64-bit shift buffer
  reg [63:0] shbuf;
  reg [5:0]  bits_cnt;
  // simple FIFO fill
  always @(posedge clk) begin
    if (!rst_n) begin
      shbuf <= 64'd0; bits_cnt <= 0; in_ready <= 1'b1;
      req_data <= 0; req_ready <= 0; ue_value <= 0; ue_ready <= 0;
      huff_sym <= 0; huff_valid <= 0;
    end else begin
      // accept input bytes when room
      if (in_valid && in_ready) begin
        shbuf <= {shbuf[55:0], in_byte}; bits_cnt <= bits_cnt + 8;
        in_ready <= (bits_cnt + 8 <= 56);
      end else if (in_valid) in_ready <= 0;
      // process read_bits
      if (req_valid && !req_ready && bits_cnt >= req_bits) begin
        req_data <= shbuf[63 -: 32] >> (32 - req_bits); // MSB-first extract
        shbuf <= shbuf << req_bits; bits_cnt <= bits_cnt - req_bits;
        req_ready <= 1'b1;
      end else req_ready <= 1'b0;
      // exp-Golomb unsigned decode (UE)
      if (req_ue_valid && !ue_ready) begin
        integer k; integer leading;
        leading = 0;
        // count leading zeros (simple loop; synthesis-friendly for small width)
        for (k=0;k<32;k=k+1) if (shbuf[63-k]==1'b0) leading = leading+1; else begin k=32; end
        // ensure bits available
        if (bits_cnt >= (2*leading+1)) begin
          ue_value <= ( (shbuf >> (63-(2*leading))) & ((1<<(leading))-1) ) + (1<= 16) begin
        reg [15:0] prefix;
        prefix = shbuf[63:48];
        // example table: if prefix top nibble equals 4'b1110 -> symbol 5 (placeholder)
        if (prefix[15:12]==4'b1110) begin
          huff_sym <= 16'd5; huff_valid <= 1'b1;
          shbuf <= shbuf << 4; bits_cnt <= bits_cnt - 4;
        end else huff_valid <= 1'b0;
      end else huff_valid <= 1'b0;
    end
  end
endmodule