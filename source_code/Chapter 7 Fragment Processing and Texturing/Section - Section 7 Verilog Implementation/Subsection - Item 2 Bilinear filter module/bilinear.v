module bilinear_filter #(
  parameter CHANNELS = 4,             // RGBA
  parameter CH_W = 8,                 // bits per channel
  parameter FRACT_BITS = 8            // fractional precision for f_x,f_y (Q0.F)
)(
  input clk, input rst,
  // handshake and inputs
  input in_valid, output reg in_ready,
  input [CHANNELS*CH_W-1:0] texel00, // packed channels MSB..LSB
  input [CHANNELS*CH_W-1:0] texel10,
  input [CHANNELS*CH_W-1:0] texel01,
  input [CHANNELS*CH_W-1:0] texel11,
  input [FRACT_BITS-1:0] fx, fy,     // fractional offsets
  // output handshake and pixel
  output reg out_valid, input out_ready,
  output reg [CHANNELS*CH_W-1:0] out_pixel
);

  localparam INT_W = CH_W + FRACT_BITS; // intermediate width

  // unpack channels
  function [CH_W-1:0] get_chan(input [CHANNELS*CH_W-1:0] pix, input integer i);
    get_chan = pix[(i+1)*CH_W-1 -: CH_W];
  endfunction

  integer i;
  // pipeline regs
  reg [INT_W-1:0] h0 [0:CHANNELS-1];
  reg [INT_W-1:0] h1 [0:CHANNELS-1];
  reg [INT_W+FRACT_BITS-1:0] v  [0:CHANNELS-1];

  // simple two-stage pipeline: horizontal then vertical
  always @(posedge clk) begin
    if (rst) begin
      in_ready <= 1;
      out_valid <= 0;
      out_pixel <= 0;
    end else begin
      // Accept new input when ready and valid
      if (in_ready && in_valid) begin
        // compute horizontal lerps: h = (1-fx)*a + fx*b  in Q0.F -> scale channels
        for (i=0;i> FRACT_BITS;
          h1[i] <= ( ( {get_chan(texel01,i), {FRACT_BITS{1'b0}}} * ({1'b0, {FRACT_BITS{1'b1}}} - fx) )
                   + ( {get_chan(texel11,i), {FRACT_BITS{1'b0}}} * fx ) ) >> FRACT_BITS;
        end
        in_ready <= 0; // consume until pipeline advances
      end

      // second stage: vertical blend when horizontal results available
      if (!in_ready) begin
        for (i=0;i> FRACT_BITS;
          // saturate and pack to output when out_ready
          if (out_ready || !out_valid) begin
            out_pixel[(i+1)*CH_W-1 -: CH_W] <= v[i][INT_W-1 -: CH_W]; // truncate high bits
          end
        end
        out_valid <= 1;
        in_ready <= 1; // ready for next input (throughput 1/cycle)
      end

      if (out_valid && out_ready) out_valid <= 0;
    end
  end

endmodule