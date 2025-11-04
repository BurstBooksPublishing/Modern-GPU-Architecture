module thermal_monitor #(
  parameter N = 8,               // number of sensors
  parameter W = 12               // temperature bitwidth
)(
  input  wire clk,
  input  wire rstn,
  input  wire [N*W-1:0] temp_bus, // concatenated temps MSB-first
  input  wire [W-1:0] th_warn,    // warning threshold
  input  wire [W-1:0] th_crit,    // critical threshold
  output reg  warn,               // warn signal
  output reg  crit,               // critical signal
  output reg  [$clog2(N)-1:0] idx_max // index of hottest sensor
);
  integer i;
  reg [W-1:0] max_t;
  always @(posedge clk or negedge rstn) begin
    if(!rstn) begin
      max_t <= {W{1'b0}}; warn <= 1'b0; crit <= 1'b0; idx_max <= 0;
    end else begin
      max_t <= {W{1'b0}};
      idx_max <= 0;
      for(i=0;i max_t) begin
          max_t <= temp_bus[(N-1-i)*W +: W];
          idx_max <= i[$clog2(N)-1:0];
        end
      end
      warn <= (max_t >= th_warn);
      crit <= (max_t >= th_crit);
    end
  end
endmodule