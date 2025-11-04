module ucie_credit_counter #(parameter WIDTH=8) (
    input  wire             clk,
    input  wire             rstn,
    input  wire             credit_inc, // +1 credit event (rx grant)
    input  wire             credit_dec, // -1 credit on flit sent
    output reg  [WIDTH-1:0] credits,    // current credits
    output wire             credit_empty // true if no credits
);
    // synchronous credit update, saturates at max value
    always @(posedge clk) begin
        if (!rstn) credits <= {WIDTH{1'b0}};
        else begin
            case ({credit_inc, credit_dec})
                2'b10: credits <= credits + 1;               // increase
                2'b01: credits <= (credits != 0) ? credits - 1 : 0; // consume
                default: credits <= credits;                 // no change or both
            endcase
        end
    end
    assign credit_empty = (credits == 0);
endmodule