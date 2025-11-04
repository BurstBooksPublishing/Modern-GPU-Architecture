module aabb_intersect #(
  parameter W = 32 // Q16.16 fixed-point
)(
  input  wire                clk,
  input  wire                rst_n,
  input  wire                valid_in,
  input  wire signed [W-1:0] ox, oy, oz,       // origin
  input  wire signed [W-1:0] inv_dx, inv_dy, inv_dz, // precomputed inv_dir
  input  wire signed [W-1:0] bx0, by0, bz0,    // box min
  input  wire signed [W-1:0] bx1, by1, bz1,    // box max
  output reg                 valid_out,
  output reg                 hit,
  output reg signed [W-1:0]  t_near
);
  // pipeline registers
  reg signed [W-1:0] t1x, t2x, t1y, t2y, t1z, t2z;
  reg signed [W-1:0] mnx, mxx, mny, mxy, mnz, mxz;
  reg signed [W-1:0] tmin, tmax;

  always @(posedge clk) begin
    if (!rst_n) begin
      valid_out <= 0; hit <= 0; t_near <= 0;
      {t1x,t2x,t1y,t2y,t1z,t2z} <= 0;
    end else begin
      if (valid_in) begin
        t1x <= (bx0 - ox) * inv_dx >>> 16; // Q16.16 mult
        t2x <= (bx1 - ox) * inv_dx >>> 16;
        t1y <= (by0 - oy) * inv_dy >>> 16;
        t2y <= (by1 - oy) * inv_dy >>> 16;
        t1z <= (bz0 - oz) * inv_dz >>> 16;
        t2z <= (bz1 - oz) * inv_dz >>> 16;
        valid_out <= 1;
      end else begin
        valid_out <= 0;
      end

      // compute mins/maxes and overlap check one cycle later
      mnx <= (t1x < t2x) ? t1x : t2x; mxx <= (t1x < t2x) ? t2x : t1x;
      mny <= (t1y < t2y) ? t1y : t2y; mxy <= (t1y < t2y) ? t2y : t1y;
      mnz <= (t1z < t2z) ? t1z : t2z; mxz <= (t1z < t2z) ? t2z : t1z;
      tmin <= (mnx > mny) ? ((mnx > mnz) ? mnx : mnz) : ((mny > mnz) ? mny : mnz);
      tmax <= (mxx < mxy) ? ((mxx < mxz) ? mxx : mxz) : ((mxy < mxz) ? mxy : mxz);
      hit <= (tmax >= ((tmin>0) ? tmin : 0));
      t_near <= tmin;
    end
  end
endmodule

module rt_core_harness (
  input wire clk,
  input wire rst_n,
  output reg [31:0] hit_count,
  output reg [31:0] miss_count,
  output reg error_flag
);
  // simple test: three rays against box [0.5,0.5,0.5]-[1.5,1.5,1.5]
  reg valid_in;
  reg signed [31:0] ox,oy,oz, inv_dx,inv_dy,inv_dz;
  reg [2:0] seq;
  wire valid_out, hit;
  wire signed [31:0] t_near;

  aabb_intersect dut(
    .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
    .ox(ox), .oy(oy), .oz(oz), .inv_dx(inv_dx), .inv_dy(inv_dy), .inv_dz(inv_dz),
    .bx0(32'h00008000), .by0(32'h00008000), .bz0(32'h00008000), // 0.5 Q16.16
    .bx1(32'h00018000), .by1(32'h00018000), .bz1(32'h00018000), // 1.5 Q16.16
    .valid_out(valid_out), .hit(hit), .t_near(t_near)
  );

  always @(posedge clk) begin
    if (!rst_n) begin
      seq <= 0; valid_in <= 0; hit_count <= 0; miss_count <= 0; error_flag <= 0;
    end else begin
      case (seq)
        0: begin // ray towards box center (hit)
          ox <= 32'h00000000; oy <= 32'h00000000; oz <= 32'h00000000; // origin 0
          inv_dx <= 32'h00010000; inv_dy <= 32'h00010000; inv_dz <= 32'h00010000; // dir=1 => inv=1
          valid_in <= 1; seq <= 1;
        end
        1: begin valid_in <= 0; if (valid_out) begin if (hit) hit_count <= hit_count+1; else miss_count <= miss_count+1; seq <= 2; end end
        2: begin // ray away from box (miss)
          ox <= 32'h00000000; oy <= 32'h00000000; oz <= 32'h00000000;
          inv_dx <= -32'sh00010000; inv_dy <= -32'sh00010000; inv_dz <= -32'sh00010000; // dir=-1
          valid_in <= 1; seq <= 3;
        end
        3: begin valid_in <= 0; if (valid_out) begin if (hit) error_flag <= 1; else miss_count <= miss_count+1; seq <= 4; end end
        4: seq <= 4; // stop
      endcase
    end
  end
endmodule