module rasterizer_core #(
  parameter XW = 10, // pixel coord width
  parameter AW = 22, // edge coeff width (signed)
  parameter ZW = 32, // depth fixed-point width
  parameter FB = 16  // fraction bits
) (
  input  wire                  clk,
  input  wire                  rstn,
  // triangle setup inputs (valid handshake)
  input  wire                  in_valid,
  output reg                   in_ready,
  input  wire signed [AW-1:0]  A0, B0, C0, // edge0 coeffs
  input  wire signed [AW-1:0]  A1, B1, C1, // edge1
  input  wire signed [AW-1:0]  A2, B2, C2, // edge2
  input  wire [XW-1:0]         min_x, min_y, max_x, max_y,
  input  wire signed [ZW-1:0]  Gx_z, Gy_z, Gc_z, // depth gradients
  // pixel output (valid-ready)
  output reg                   out_valid,
  input  wire                  out_ready,
  output reg [XW-1:0]          px, py,
  output reg signed [ZW-1:0]   depth,
  output reg [2:0]             covered
);

  // internal registers
  reg [XW-1:0] cur_x, cur_y;
  reg signed [AW-1:0] e0_row, e1_row, e2_row; // row-start edge values
  reg signed [AW-1:0] e0, e1, e2; // per-pixel edge values
  reg signed [ZW-1:0] z_row, z_pix;

  // state machine: IDLE -> RASTER -> DONE
  localparam IDLE=0, RASTER=1;
  reg state;

  // accept inputs
  always @(posedge clk) begin
    if (!rstn) begin
      in_ready <= 1'b1;
      state <= IDLE;
      out_valid <= 1'b0;
    end else begin
      if (state==IDLE && in_valid && in_ready) begin
        // initialize walker at min_x,min_y
        cur_x <= min_x;
        cur_y <= min_y;
        // compute row-start edge values: E(min_x,min_y) = A*min_x + B*min_y + C
        e0_row <= A0*min_x + B0*min_y + C0;
        e1_row <= A1*min_x + B1*min_y + C1;
        e2_row <= A2*min_x + B2*min_y + C2;
        // depth row-start
        z_row <= (Gx_z*min_x + Gy_z*min_y + Gc_z);
        // load per-pixel from row-start
        e0 <= A0*min_x + B0*min_y + C0;
        e1 <= A1*min_x + B1*min_y + C1;
        e2 <= A2*min_x + B2*min_y + C2;
        z_pix <= (Gx_z*min_x + Gy_z*min_y + Gc_z);
        in_ready <= 1'b0;
        state <= RASTER;
        out_valid <= 1'b0;
      end else if (state==RASTER) begin
        if (!out_valid && out_ready) begin end
        // drive output if pixel within bbox
        if (!out_valid) begin
          px <= cur_x;
          py <= cur_y;
          depth <= z_pix;
          covered <= {e0>=0, e1>=0, e2>=0};
          out_valid <= 1'b1;
        end else if (out_valid && out_ready) begin
          out_valid <= 1'b0;
          // advance X
          if (cur_x < max_x) begin
            cur_x <= cur_x + 1;
            // incremental edge updates for X step
            e0 <= e0 + A0;
            e1 <= e1 + A1;
            e2 <= e2 + A2;
            z_pix <= z_pix + Gx_z;
          end else begin
            // move to next row
            if (cur_y < max_y) begin
              cur_y <= cur_y + 1;
              cur_x <= min_x;
              // row-start edges: add B to previous row-start
              e0_row <= e0_row + B0;
              e1_row <= e1_row + B1;
              e2_row <= e2_row + B2;
              e0 <= e0_row + B0; // set to new row-start
              e1 <= e1_row + B1;
              e2 <= e2_row + B2;
              z_row <= z_row + Gy_z;
              z_pix <= z_row + Gy_z;
            end else begin
              // finished triangle
              state <= IDLE;
              in_ready <= 1'b1;
              out_valid <= 1'b0;
            end
          end
        end
      end
    end
  end

endmodule