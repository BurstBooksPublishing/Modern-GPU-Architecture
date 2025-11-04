module perspective_division_q16 (
  input  wire         clk,
  input  wire         rst_n,
  input  wire         valid_in,
  input  wire signed [31:0] x_in, // Q16.16
  input  wire signed [31:0] y_in, // Q16.16
  input  wire signed [31:0] z_in, // Q16.16
  input  wire signed [31:0] w_in, // Q16.16
  output reg          valid_out,
  output reg signed [31:0] x_ndc, // Q16.16
  output reg signed [31:0] y_ndc, // Q16.16
  output reg signed [31:0] z_ndc, // Q16.16
  output reg signed [31:0] inv_w  // Q16.16 = 1/w
);
  localparam integer FRAC = 16;
  // Pipeline: compute 1/w then multiply via shifted division
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
      x_ndc <= 32'sd0; y_ndc <= 32'sd0; z_ndc <= 32'sd0; inv_w <= 32'sd0;
    end else begin
      if (valid_in && (w_in != 0)) begin
        // inv_w = (1 << FRAC) / w_in  in Q16.16 (signed division)
        inv_w <= ($signed(32'sd1 <<< FRAC) <<< 0) / $signed(w_in);
        // NDC = (coord << FRAC) / w_in
        x_ndc <= ($signed(x_in) <<< FRAC) / $signed(w_in);
        y_ndc <= ($signed(y_in) <<< FRAC) / $signed(w_in);
        z_ndc <= ($signed(z_in) <<< FRAC) / $signed(w_in);
        valid_out <= 1'b1;
      end else begin
        valid_out <= 1'b0;
      end
    end
  end
endmodule