module bandwidth_calc #(
  parameter INW = 64, // input width
  parameter OUTW = 128 // output width for product
)(
  input  wire [31:0]  width_bits,       // bits per channel
  input  wire [INW-1:0] transfers_per_s, // transfers per second (Hz)
  input  wire [15:0]  num_channels,
  output wire [OUTW-1:0] bytes_per_s
);
  // Compute bits_per_s = width_bits * transfers_per_s * num_channels
  wire [OUTW-1:0] bits_per_s;
  assign bits_per_s = $signed(width_bits) * $signed(transfers_per_s) * $signed(num_channels);
  // divide by 8 to get bytes/s
  assign bytes_per_s = bits_per_s / 8;
endmodule