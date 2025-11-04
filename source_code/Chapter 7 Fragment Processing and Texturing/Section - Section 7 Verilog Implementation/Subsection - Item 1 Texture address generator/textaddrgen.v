module texture_addr_gen #(
  parameter FIX = 16,                // fractional bits
  parameter MAX_W = 4096,            // power-of-two
  parameter MAX_H = 4096,
  parameter LOD_BITS = 4
)(
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire                 valid_in,
  input  wire [31:0]          u_in,        // Q16.16 normalized
  input  wire [31:0]          v_in,        // Q16.16 normalized
  input  wire [LOD_BITS-1:0]  lod_in,      // integer LOD
  input  wire [1:0]           wrap_u,      // 00:clamp 01:repeat 10:mirror
  input  wire [1:0]           wrap_v,
  input  wire [13:0]          base_width,  // power-of-two
  input  wire [13:0]          base_height,
  output reg                  valid_out,
  output reg  [13:0]          tex_x,
  output reg  [13:0]          tex_y,
  output reg  [31:0]          addr_out,    // linear index (y*stride + x)
  output reg  [LOD_BITS-1:0]  mip_out
);

  // compute level dimension with min 1
  wire [13:0] level_w = (base_width >> lod_in) ? (base_width >> lod_in) : 14'd1;
  wire [13:0] level_h = (base_height >> lod_in) ? (base_height >> lod_in) : 14'd1;

  // multiply Q16.16 * size -> wide, then shift down FIX
  wire [47:0] mult_x = u_in * level_w;
  wire [47:0] mult_y = v_in * level_h;
  wire [31:0] pos_x_fp = mult_x >> FIX; // integer pixel position, may exceed size
  wire [31:0] pos_y_fp = mult_y >> FIX;

  // masks (power-of-two requirement)
  wire [13:0] mask_w = level_w - 1;
  wire [13:0] mask_h = level_h - 1;

  // combinational wrap logic
  reg [13:0] x_raw, y_raw;
  always @(*) begin
    // wrap for X
    case (wrap_u)
      2'b01: begin // repeat: modulo via mask
        x_raw = pos_x_fp[13:0] & mask_w;
      end
      2'b10: begin // mirror repeat
        // pos mod (2*W)
        wire [14:0] mod2 = pos_x_fp[14:0] & ((level_w<<1)-1);
        if (mod2 >= level_w) x_raw = ((level_w<<1)-1) - mod2;
        else x_raw = mod2[13:0];
      end
      default: begin // clamp
        if (pos_x_fp[31:0] >= level_w) x_raw = level_w - 1;
        else x_raw = pos_x_fp[13:0];
      end
    endcase
    // wrap for Y
    case (wrap_v)
      2'b01: y_raw = pos_y_fp[13:0] & mask_h;
      2'b10: begin
        wire [14:0] m2 = pos_y_fp[14:0] & ((level_h<<1)-1);
        if (m2 >= level_h) y_raw = ((level_h<<1)-1) - m2;
        else y_raw = m2[13:0];
      end
      default: begin
        if (pos_y_fp[31:0] >= level_h) y_raw = level_h - 1;
        else y_raw = pos_y_fp[13:0];
      end
    endcase
  end

  // pipeline registers for single-cycle throughput into TMU
  reg [13:0] tex_x_r, tex_y_r;
  reg [31:0] addr_r;
  reg [LOD_BITS-1:0] mip_r;
  reg valid_r;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_r <= 1'b0;
      tex_x_r <= 14'd0; tex_y_r <= 14'd0; addr_r <= 32'd0; mip_r <= {LOD_BITS{1'b0}};
    end else begin
      valid_r <= valid_in;
      tex_x_r <= x_raw;
      tex_y_r <= y_raw;
      mip_r <= lod_in;
      addr_r <= (y_raw * level_w) + x_raw; // linear index; stride == level_w
    end
  end

  // outputs
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
      tex_x <= 14'd0; tex_y <= 14'd0; addr_out <= 32'd0; mip_out <= {LOD_BITS{1'b0}};
    end else begin
      valid_out <= valid_r;
      tex_x <= tex_x_r;
      tex_y <= tex_y_r;
      addr_out <= addr_r;
      mip_out <= mip_r;
    end
  end

endmodule