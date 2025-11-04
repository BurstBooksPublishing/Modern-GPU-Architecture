module tex_addr_wrap #(
  parameter INT = 4,            // integer bits of input
  parameter FP  = 16,           // fractional bits of input
  parameter W   = 12            // bits for texture size (max size = 2^W)
)(
  input  wire                     clk,
  input  wire                     rst_n,
  input  wire signed [(INT+FP)-1:0] u_in, // signed Q(INT).Q(FP)
  input  wire signed [(INT+FP)-1:0] v_in,
  input  wire [W-1:0]             width,  // texture width (texels)
  input  wire [W-1:0]             height,
  input  wire [1:0]               mode_u, // 0=repeat,1=clamp,2=mirror,3=border
  input  wire [1:0]               mode_v,
  output reg  [W-1:0]             x_out,
  output reg  [W-1:0]             y_out,
  output reg                      border_out
);

  // local signed scaling intermediate (sized to avoid overflow)
  localparam SCALEW = INT + FP + W;
  wire signed [SCALEW-1:0] su = (u_in * $signed({1'b0, width})) >>> FP;
  wire signed [SCALEW-1:0] sv = (v_in * $signed({1'b0, height})) >>> FP;

  // helper function implemented combinationally via always block
  function [W-1:0] resolve_idx;
    input signed [SCALEW-1:0] s;
    input [W-1:0] size;
    input [1:0] mode;
    reg signed [SCALEW+1:0] t;
    reg signed [SCALEW+1:0] period;
    begin
      resolve_idx = {W{1'b0}};
      case (mode)
        2'd0: begin // repeat
          if (size == 0) resolve_idx = {W{1'b0}};
          else begin
            // modulo positive remainder
            t = s % $signed({{(SCALEW+2-W){1'b0}}, size});
            if (t < 0) t = t + $signed({{(SCALEW+2-W){1'b0}}, size});
            resolve_idx = t[W-1:0];
          end
        end
        2'd1: begin // clamp-to-edge
          if (s < 0) resolve_idx = {W{1'b0}};
          else if (s >= $signed({{(SCALEW+2-W){1'b0}}, size})) resolve_idx = size - 1;
          else resolve_idx = s[W-1:0];
        end
        2'd2: begin // mirror
          period = $signed({{(SCALEW+2-W){1'b0}}, size}) * 2;
          if (period == 0) resolve_idx = {W{1'b0}};
          else begin
            t = s % period;
            if (t < 0) t = t + period;
            if (t >= $signed({{(SCALEW+2-W){1'b0}}, size}))
              resolve_idx = (period - 1 - t)[W-1:0];
            else resolve_idx = t[W-1:0];
          end
        end
        2'd3: begin // clamp-to-border, caller checks border
          if (s < 0) resolve_idx = {W{1'b0}};
          else if (s >= $signed({{(SCALEW+2-W){1'b0}}, size})) resolve_idx = size - 1;
          else resolve_idx = s[W-1:0];
        end
      endcase
    end
  endfunction

  // border detection combinationally
  wire u_oob = (su < 0) || (su >= $signed({{(SCALEW+2-W){1'b0}}, width}));
  wire v_oob = (sv < 0) || (sv >= $signed({{(SCALEW+2-W){1'b0}}, height}));

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_out <= 0;
      y_out <= 0;
      border_out <= 0;
    end else begin
      x_out <= resolve_idx(su, width, mode_u);
      y_out <= resolve_idx(sv, height, mode_v);
      border_out <= (mode_u == 2'd3 && u_oob) || (mode_v == 2'd3 && v_oob); // border if either coord outside
    end
  end

endmodule