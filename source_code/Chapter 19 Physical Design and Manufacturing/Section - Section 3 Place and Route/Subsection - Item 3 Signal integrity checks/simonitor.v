module si_activity_monitor #(
  parameter WIDTH = 1,         // number of monitored nets
  parameter CNT_BITS = 16,
  parameter THRESH = 1000      // toggle threshold
)(
  input  wire clk,
  input  wire rst_n,
  input  wire [WIDTH-1:0] sig, // sample of net(s)
  output reg  [WIDTH-1:0] alert // sticky alert per net
);
  reg [WIDTH-1:0] prev;
  reg [CNT_BITS-1:0] cnt [WIDTH-1:0];
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prev <= {WIDTH{1'b0}};
      alert <= {WIDTH{1'b0}};
      for (i=0;i= THRESH) alert[i] <= 1'b1; // flag high activity
        end
        prev[i] <= sig[i];
      end
    end
  end
endmodule