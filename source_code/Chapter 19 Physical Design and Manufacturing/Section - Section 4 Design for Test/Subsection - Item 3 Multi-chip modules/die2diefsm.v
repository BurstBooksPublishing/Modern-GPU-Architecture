module die2die_link #(parameter CREDITS=8) (
  input  wire clk, rstn,            // clock and active-low reset
  input  wire tx_req,               // request to send (from producer)
  output reg  tx_ack,               // link grants send
  input  wire rx_credit,            // incoming credit returned
  output reg  rx_consume            // consume signal to receiver
);
  reg [$clog2(CREDITS):0] credits;   // credit counter
  // initialize credits on reset
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      credits <= CREDITS;
      tx_ack <= 1'b0;
      rx_consume <= 1'b0;
    end else begin
      // grant if credits available
      if (tx_req && credits > 0) begin
        tx_ack <= 1'b1;
        credits <= credits - 1;
      end else begin
        tx_ack <= 1'b0;
      end
      // return credit when peer signals consumption
      if (rx_credit) credits <= credits + 1;
      // consumer pulse (example) - could be handshake-driven
      rx_consume <= tx_ack; // simple loopback consume for pipeline
    end
  end
endmodule