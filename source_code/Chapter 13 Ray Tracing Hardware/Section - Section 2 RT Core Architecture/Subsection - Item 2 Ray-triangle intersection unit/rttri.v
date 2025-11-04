module rt_tri_intf #(parameter W=32, F=30) (
  input  wire                 clk,
  input  wire                 rstn,
  // input ray (origin, direction)
  input  wire                 in_valid,
  input  wire [W-1:0]         ox, oy, oz,
  input  wire [W-1:0]         dx, dy, dz,
  // triangle vertices
  input  wire [W-1:0]         v0x, v0y, v0z,
  input  wire [W-1:0]         v1x, v1y, v1z,
  input  wire [W-1:0]         v2x, v2y, v2z,
  input  wire                 backface_cull,
  output reg                  out_valid,
  output reg  [W-1:0]         tout, uout, vout,
  output reg                  hit
);
  // Stage 1: edge vectors
  wire signed [W-1:0] e1x = v1x - v0x;
  wire signed [W-1:0] e1y = v1y - v0y;
  wire signed [W-1:0] e1z = v1z - v0z;
  wire signed [W-1:0] e2x = v2x - v0x;
  wire signed [W-1:0] e2y = v2y - v0y;
  wire signed [W-1:0] e2z = v2z - v0z;
  // Stage 2: p = d x e2 (fixed-point mults)
  wire signed [W-1:0] px = (dy*e2z - dz*e2y) >>> F;
  wire signed [W-1:0] py = (dz*e2x - dx*e2z) >>> F;
  wire signed [W-1:0] pz = (dx*e2y - dy*e2x) >>> F;
  // det = e1 . p
  wire signed [W-1:0] det = (e1x*px + e1y*py + e1z*pz) >>> F;
  // tvec = o - v0
  wire signed [W-1:0] tx = ox - v0x;
  wire signed [W-1:0] ty = oy - v0y;
  wire signed [W-1:0] tz = oz - v0z;
  // u numerator
  wire signed [W-1:0] un = (tx*px + ty*py + tz*pz) >>> F;
  // q = t x e1
  wire signed [W-1:0] qx = (ty*e1z - tz*e1y) >>> F;
  wire signed [W-1:0] qy = (tz*e1x - tx*e1z) >>> F;
  wire signed [W-1:0] qz = (tx*e1y - ty*e1x) >>> F;
  // v numerator and t numerator
  wire signed [W-1:0] vn = (dx*qx + dy*qy + dz*qz) >>> F;
  wire signed [W-1:0] tn = (e2x*qx + e2y*qy + e2z*qz) >>> F;
  // simple inverse approximation: use reciprocal via Newton iteration externally.
  // Here compute hit tests without explicit division by comparing scaled ranges.
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin out_valid<=0; hit<=0; end
    else begin
      out_valid <= in_valid; // one-cycle flow model for example
      // culling and epsilon check
      if (in_valid) begin
        if (det == 0) begin hit<=0; end
        else begin
          // backface cull
          if (backface_cull && det < 0) hit<=0;
          else begin
            // compare u = un/det in [0,1] via sign and magnitude cross-multiplication
            if ((un < 0) || (un > det)) hit<=0;
            else if ((vn < 0) || (un + vn > det)) hit<=0;
            else if (tn <= 0) hit<=0;
            else begin
              hit <= 1;
              // approximate outputs (fixed-point division omitted for brevity)
              tout <= tn; uout <= un; vout <= vn;
            end
          end
        end
      end else hit<=0;
    end
  end
endmodule