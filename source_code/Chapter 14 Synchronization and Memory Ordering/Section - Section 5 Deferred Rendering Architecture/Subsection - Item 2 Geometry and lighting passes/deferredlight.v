module deferred_lighting_core (
  input  wire         clk,
  input  wire         rst_n,
  input  wire         valid_in,           // input sample valid
  input  wire [15:0]  albedo_r, albedo_g, albedo_b, // Q1.15
  input  wire [15:0]  nx, ny, nz,         // normal Q1.15
  input  wire [15:0]  lx, ly, lz,         // light dir Q1.15 (normalized)
  input  wire [7:0]   specular_scale,     // 0..255 mapped to Q1.7
  input  wire [3:0]   shininess,          // small integer exponent
  output reg  [15:0]  out_r, out_g, out_b, // Q1.15
  output reg          valid_out
);
  // dot = nx*lx + ny*ly + nz*lz  -> Q2.30 sum, then downshift 15 -> Q1.15
  wire signed [31:0] mul_nl_x = $signed(nx) * $signed(lx);
  wire signed [31:0] mul_nl_y = $signed(ny) * $signed(ly);
  wire signed [31:0] mul_nl_z = $signed(nz) * $signed(lz);
  wire signed [33:0] dot_sum = $signed(mul_nl_x) + $signed(mul_nl_y) + $signed(mul_nl_z);
  wire signed [15:0] dot_nl = dot_sum[30:15]; // take Q1.15, simple truncation

  // clamp to >=0
  wire signed [15:0] nld = (dot_nl[15] ? 16'd0 : dot_nl);

  // compute half-vector h = normalize(l+v). For deferred, approximate view=(0,0,1)
  wire signed [15:0] vx = 16'sd0, vy = 16'sd0, vz = 16'sd16384; // Q1.15 = 1.0
  wire signed [15:0] hx = lx + vx;
  wire signed [15:0] hy = ly + vy;
  wire signed [15:0] hz = lz + vz;
  // skip normalization; compute n.h as dot with unnormalized h and scale later
  wire signed [31:0] mul_nh_x = $signed(nx) * $signed(hx);
  wire signed [31:0] mul_nh_y = $signed(ny) * $signed(hy);
  wire signed [31:0] mul_nh_z = $signed(nz) * $signed(hz);
  wire signed [33:0] dot_h_sum = $signed(mul_nh_x) + $signed(mul_nh_y) + $signed(mul_nh_z);
  wire signed [15:0] dot_nh_raw = dot_h_sum[30:15];
  wire signed [15:0] ndh = (dot_nh_raw[15] ? 16'd0 : dot_nh_raw);

  // Specular power via repeated multiply (integer exponent)
  reg signed [31:0] spec_acc;
  integer i;
  always @(*) begin
    spec_acc = ndh; // Q1.15
    for (i=1;i>> 15; // keep Q1.15
  end

  // Combine: diffuse = albedo * nld; spec = specular_scale * spec_acc
  wire signed [31:0] diffr = ($signed(albedo_r) * $signed(nld)) >>> 15;
  wire signed [31:0] specr = ($signed({8'd0,specular_scale}) * spec_acc) >>> 7; // scale Q1.7
  // clamp and register output
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_out <= 1'b0; out_r <= 0; out_g <= 0; out_b <= 0;
    end else begin
      valid_out <= valid_in;
      if (valid_in) begin
        out_r <= (diffr + specr > 16'sd32767) ? 16'sd32767 : (diffr + specr);
        out_g <= ( ($signed(albedo_g) * $signed(nld))>>>15 ); // simpler channels
        out_b <= ( ($signed(albedo_b) * $signed(nld))>>>15 );
      end
    end
  end
endmodule