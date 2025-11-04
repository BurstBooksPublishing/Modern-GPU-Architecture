module timing_window_gen #(
  parameter integer LATENCY = 3,      // cycles to wait
  parameter integer WIDTH   = 1       // strobe width in cycles
)(
  input  wire        clk,             // system clock
  input  wire        rst_n,           // active-low reset
  input  wire        start,           // start pulse (one cycle)
  output wire        strobe           // generated strobe (WIDTH cycles)
);
  // counter counts down once start seen; value 0 => strobe asserted
  reg [$clog2(LATENCY+1)-1:0] cnt;
  reg active;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cnt    <= 0;
      active <= 1'b0;
    end else begin
      if (start && !active) begin
        cnt    <= LATENCY;        // capture latency
        active <= 1'b1;
      end else if (active) begin
        if (cnt == 0) begin
          active <= 1'b0;
        end else begin
          cnt <= cnt - 1;
        end
      end
    end
  end
  assign strobe = (active && (cnt == 0));
endmodule