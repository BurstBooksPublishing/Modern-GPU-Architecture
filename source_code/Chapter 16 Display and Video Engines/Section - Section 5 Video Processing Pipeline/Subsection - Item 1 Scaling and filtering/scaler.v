module scaler_bilinear #(
  parameter IN_WIDTH = 1920,
  parameter FRAC = 16
)(
  input  wire         clk,
  input  wire         rst,
  input  wire [23:0]  in_pixel,    // {R[23:16],G[15:8],B[7:0]}
  input  wire         in_valid,
  input  wire [31:0]  scale_x,     // 16.16 fixed-point
  input  wire [31:0]  scale_y,     // 16.16 fixed-point
  output reg  [23:0]  out_pixel,
  output reg          out_valid
);
  // line buffers for two most recent lines
  reg [23:0] line0 [0:IN_WIDTH-1];
  reg [23:0] line1 [0:IN_WIDTH-1];
  integer write_x;
  integer line_write_ptr;
  reg toggle; // 0: line0 is older, 1: line1 older

  // phase accumulators for output sampling
  reg [31:0] phase_x;
  reg [31:0] phase_y;
  reg [31:0] phase_x_step;
  reg [31:0] phase_y_step;

  // simple state: fill first two lines, then produce outputs
  reg [1:0] line_count;

  // initialize
  always @(posedge clk) begin
    if (rst) begin
      write_x <= 0;
      line_write_ptr <= 0;
      toggle <= 0;
      phase_x <= 0;
      phase_y <= 0;
      phase_x_step <= scale_x;
      phase_y_step <= scale_y;
      line_count <= 0;
      out_valid <= 0;
    end else begin
      phase_x_step <= scale_x;
      phase_y_step <= scale_y;
      if (in_valid) begin
        // write incoming pixel into the current write line
        if (toggle == 0) line0[write_x] <= in_pixel;
        else             line1[write_x] <= in_pixel;
        write_x <= write_x + 1;
        if (write_x == IN_WIDTH-1) begin
          write_x <= 0;
          line_count <= (line_count<2) ? line_count+1 : 2;
          toggle <= ~toggle; // advance ring
        end
      end

      // produce output only when at least two lines buffered
      if (line_count >= 2) begin
        // sample indices and fractional parts
        // ix = phase_x >> FRAC, fx = lower FRAC bits
        integer ix; integer iy;
        reg [FRAC-1:0] fx, fy;
        reg [23:0] p00, p10, p01, p11;
        reg [23:0] h0, h1;
        ix = phase_x[31:FRAC]; fx = phase_x[FRAC-1:0];
        iy = phase_y[31:FRAC]; fy = phase_y[FRAC-1:0];

        // clamp indices
        if (ix >= IN_WIDTH-1) ix = IN_WIDTH-1;
        // choose lines: top and bottom (older/newer depending on toggle)
        if (toggle == 0) begin
          p00 = line1[ix];      // top
          p10 = line1[(ix==IN_WIDTH-1)?ix:ix+1];
          p01 = line0[ix];      // bottom
          p11 = line0[(ix==IN_WIDTH-1)?ix:ix+1];
        end else begin
          p00 = line0[ix];
          p10 = line0[(ix==IN_WIDTH-1)?ix:ix+1];
          p01 = line1[ix];
          p11 = line1[(ix==IN_WIDTH-1)?ix:ix+1];
        end

        // horizontal interpolation for each channel
        // R channel
        integer r00,r10,r01,r11;
        integer rh0,rh1;
        r00 = p00[23:16]; r10 = p10[23:16];
        r01 = p01[23:16]; r11 = p11[23:16];
        rh0 = (( ( (1<<FRAC) - fx ) * r00 ) + ( fx * r10 )) >> FRAC;
        rh1 = (( ( (1<<FRAC) - fx ) * r01 ) + ( fx * r11 )) >> FRAC;
        // vertical
        integer rout;
        rout = (( ( (1<<FRAC) - fy ) * rh0 ) + ( fy * rh1 )) >> FRAC;

        // repeat for G channel
        integer g00,g10,g01,g11,gh0,gh1,gout;
        g00 = p00[15:8]; g10 = p10[15:8];
        g01 = p01[15:8]; g11 = p11[15:8];
        gh0 = ((( (1<<FRAC) - fx ) * g00 ) + ( fx * g10 )) >> FRAC;
        gh1 = ((( (1<<FRAC) - fx ) * g01 ) + ( fx * g11 )) >> FRAC;
        gout = ((( (1<<FRAC) - fy ) * gh0 ) + ( fy * gh1 )) >> FRAC;

        // B channel
        integer b00,b10,b01,b11,bh0,bh1,bout;
        b00 = p00[7:0]; b10 = p10[7:0];
        b01 = p01[7:0]; b11 = p11[7:0];
        bh0 = ((( (1<<FRAC) - fx ) * b00 ) + ( fx * b10 )) >> FRAC;
        bh1 = ((( (1<<FRAC) - fx ) * b01 ) + ( fx * b11 )) >> FRAC;
        bout = ((( (1<<FRAC) - fy ) * bh0 ) + ( fy * bh1 )) >> FRAC;

        out_pixel <= {rout[7:0], gout[7:0], bout[7:0]};
        out_valid <= 1'b1;

        // advance phases
        phase_x <= phase_x + phase_x_step;
        phase_y <= phase_y + phase_y_step;
      end else begin
        out_valid <= 1'b0;
      end
    end
  end
endmodule