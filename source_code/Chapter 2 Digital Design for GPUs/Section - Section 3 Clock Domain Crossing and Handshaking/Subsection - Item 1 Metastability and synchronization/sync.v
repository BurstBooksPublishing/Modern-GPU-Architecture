module sync_stages #(parameter STAGES = 2) (
    input  wire clk,        // destination clock (e.g., SM clock)
    input  wire din,        // asynchronous input (source domain)
    input  wire rst_n,      // synchronous active-low reset
    output wire dout        // synchronized output
);
    // shift register of flip-flops
    reg [STAGES-1:0] ff;
    always @(posedge clk) begin
        if (!rst_n) ff <= {STAGES{1'b0}};   // synchronous reset
        else        ff <= {ff[STAGES-2:0], din}; // shift in asynchronous bit
    end
    assign dout = ff[STAGES-1];
endmodule