module simt_lane_bank #(parameter LANES = 32, WIDTH = 32)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire [LANES*WIDTH-1:0]  a_flat, // concatenated inputs
    input  wire [LANES*WIDTH-1:0]  b_flat,
    output reg  [LANES*WIDTH-1:0]  sum_flat
);
    // per-lane registers and combinational adds
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : lanes
            wire [WIDTH-1:0] a = a_flat[i*WIDTH +: WIDTH];
            wire [WIDTH-1:0] b = b_flat[i*WIDTH +: WIDTH];
            wire [WIDTH-1:0] s = a + b; // lane ALU
            always @(posedge clk) begin
                if (rst) sum_flat[i*WIDTH +: WIDTH] <= {WIDTH{1'b0}};
                else       sum_flat[i*WIDTH +: WIDTH] <= s; // registered output
            end
        end
    endgenerate
endmodule