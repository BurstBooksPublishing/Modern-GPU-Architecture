module atomic_fai (
  input  wire        clk,
  input  wire        rst_n,
  input  wire        req,        // request pulse (one-hot per client should be arbitered externally)
  output reg         ack,        // pulse when value is valid
  output reg  [31:0] value       // returned prior counter value
);
  reg [31:0] counter;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      counter <= 32'd0;
      ack     <= 1'b0;
      value   <= 32'd0;
    end else
      ack <= 1'b0; // default deassert
      if (req) begin
        value   <= counter;     // capture prior value
        counter <= counter + 1; // increment for next client
        ack     <= 1'b1;        // indicate completion next cycle
      end
    end
  end
endmodule