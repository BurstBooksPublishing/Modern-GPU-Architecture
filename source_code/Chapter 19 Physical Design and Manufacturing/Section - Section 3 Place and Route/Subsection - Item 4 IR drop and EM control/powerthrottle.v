module power_throttle #(
  parameter N_SAMPS=8, parameter WIDTH=16
)(
  input  wire                 clk,
  input  wire                 rstn,
  input  wire [WIDTH-1:0]     current_sample, // instantaneous current estimate
  input  wire [7:0]           temp_degC,      // thermal input
  output reg                  throttle_req    // throttle request to SM arbiter
);
  // simple moving-window accumulator
  reg [WIDTH+3:0] acc;
  reg [3:0] idx;
  reg [WIDTH-1:0] window [0:N_SAMPS-1];
  integer i;
  wire [WIDTH+3:0] threshold = (32'd1000 + temp_degC*8); // programmable margin

  always @(posedge clk) begin
    if (!rstn) begin
      acc <= 0; idx <= 0; throttle_req <= 0;
      for (i=0;i threshold) throttle_req <= 1;
      else throttle_req <= 0;
    end
  end
endmodule