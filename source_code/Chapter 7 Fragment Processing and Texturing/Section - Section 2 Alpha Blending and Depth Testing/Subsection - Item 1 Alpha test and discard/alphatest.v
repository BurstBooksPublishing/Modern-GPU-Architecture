module alpha_test_unit (
    input  wire [7:0] alpha_frag, // normalized 0..255
    input  wire [7:0] alpha_ref,  // normalized reference
    input  wire [2:0] func,       // 3-bit compare opcode
    output reg         pass       // true => fragment survives
);
    // compare opcode encoding
    localparam FUNC_LESS   = 3'd0;
    localparam FUNC_LEQUAL = 3'd1;
    localparam FUNC_EQUAL  = 3'd2;
    localparam FUNC_NEQ    = 3'd3;
    localparam FUNC_GEQUAL = 3'd4;
    localparam FUNC_GREATER= 3'd5;

    always @* begin
        case (func)
            FUNC_LESS:    pass = (alpha_frag <  alpha_ref);
            FUNC_LEQUAL:  pass = (alpha_frag <= alpha_ref);
            FUNC_EQUAL:   pass = (alpha_frag == alpha_ref);
            FUNC_NEQ:     pass = (alpha_frag != alpha_ref);
            FUNC_GEQUAL:  pass = (alpha_frag >= alpha_ref);
            FUNC_GREATER: pass = (alpha_frag >  alpha_ref);
            default:      pass = 1'b1; // conservative default: keep fragment
        endcase
    end
endmodule