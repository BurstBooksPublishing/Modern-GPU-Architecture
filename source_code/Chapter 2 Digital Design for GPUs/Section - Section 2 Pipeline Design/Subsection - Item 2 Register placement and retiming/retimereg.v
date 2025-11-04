module retime_reg #(
  parameter WIDTH = 128
) (
  input  wire               clk,    // clock
  input  wire               rst_n,  // async reset active low
  input  wire               in_valid,
  output wire               in_ready,
  input  wire [WIDTH-1:0]   in_data,
  output wire               out_valid,
  input  wire               out_ready,
  output wire [WIDTH-1:0]   out_data
);
  // simple skid buffer to avoid combinational loop on backpressure
  reg [WIDTH-1:0]  data_q;
  reg              v_q;
  assign in_ready  = ~v_q | out_ready;            // accept if empty or downstream ready
  assign out_valid = v_q;
  assign out_data  = data_q;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      v_q   <= 1'b0;
      data_q<= {WIDTH{1'b0}};
    end else begin
      if (in_ready && in_valid) begin
        data_q <= in_data;                         // capture input when accepted
        v_q    <= 1'b1;
      end else if (out_ready && v_q) begin
        v_q    <= 1'b0;                            // consume when downstream accepts
      end
    end
  end
endmodule