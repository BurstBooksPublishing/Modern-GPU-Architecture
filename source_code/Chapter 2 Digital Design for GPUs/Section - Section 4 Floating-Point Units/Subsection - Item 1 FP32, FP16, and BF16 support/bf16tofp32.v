module bf16_to_fp32_select (
    input  wire        clk,            // optional registered path
    input  wire        rst_n,
    input  wire [31:0] fp32In,         // native FP32 input
    input  wire [15:0] bf16In,         // BF16 input (1|8|7)
    input  wire        useBf16,        // when high, convert bf16In
    output reg  [31:0] fp32Out         // widened output
);
    // BF16 fields: [15]=sign, [14:7]=exp8, [6:0]=mant7
    wire sign = bf16In[15];
    wire [7:0] exp8  = bf16In[14:7];
    wire [6:0] mant7 = bf16In[6:0];

    wire [22:0] mant32_from_bf16 = {mant7, 16'b0}; // pad low bits
    wire [30:0] abs32_from_bf16 = {exp8, mant32_from_bf16}; // exponent + mantissa

    always @(*) begin
        if (!useBf16) begin
            fp32Out = fp32In; // passthrough FP32
        end else begin
            // assemble FP32: sign|exp8|mant7<<16
            fp32Out = {sign, abs32_from_bf16};
        end
    end
endmodule