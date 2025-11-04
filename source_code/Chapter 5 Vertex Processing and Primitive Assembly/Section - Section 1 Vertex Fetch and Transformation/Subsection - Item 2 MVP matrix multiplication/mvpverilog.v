module mvp_q16_16 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [31:0]  mat [0:15], // 16 matrix elements in row-major, Q16.16
    input  wire [31:0]  vec[0:3],   // input homogeneous vector [x y z 1], Q16.16
    output reg          valid_out,
    output reg  [31:0]  out_vec[0:3] // resulting clip-space vector, Q16.16
);
    // Stage 1: parallel multiplies (signed 32x32 -> 64)
    reg signed [63:0] muls [0:15];
    reg                s_valid;
    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            s_valid <= 1'b0;
            for (i=0;i<16;i=i+1) muls[i] <= 64'd0;
        end else begin
            s_valid <= valid_in;
            // compute all 16 products
            muls[ 0] <= $signed(mat[0]) * $signed(vec[0]);
            muls[ 1] <= $signed(mat[1]) * $signed(vec[1]);
            muls[ 2] <= $signed(mat[2]) * $signed(vec[2]);
            muls[ 3] <= $signed(mat[3]) * $signed(vec[3]);
            muls[ 4] <= $signed(mat[4]) * $signed(vec[0]);
            muls[ 5] <= $signed(mat[5]) * $signed(vec[1]);
            muls[ 6] <= $signed(mat[6]) * $signed(vec[2]);
            muls[ 7] <= $signed(mat[7]) * $signed(vec[3]);
            muls[ 8] <= $signed(mat[8]) * $signed(vec[0]);
            muls[ 9] <= $signed(mat[9]) * $signed(vec[1]);
            muls[10] <= $signed(mat[10]) * $signed(vec[2]);
            muls[11] <= $signed(mat[11]) * $signed(vec[3]);
            muls[12] <= $signed(mat[12]) * $signed(vec[0]);
            muls[13] <= $signed(mat[13]) * $signed(vec[1]);
            muls[14] <= $signed(mat[14]) * $signed(vec[2]);
            muls[15] <= $signed(mat[15]) * $signed(vec[3]);
        end
    end

    // Stage 2: accumulate products into 4 outputs, then shift down Q16.16
    reg signed [95:0] acc [0:3]; // extra bits to avoid overflow
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (i=0;i<4;i=i+1) begin out_vec[i] <= 32'd0; acc[i] <= 96'd0; end
        end else begin
            valid_out <= s_valid;
            // accumulate row-wise
            acc[0] <= $signed(muls[0]) + $signed(muls[1]) + $signed(muls[2]) + $signed(muls[3]);
            acc[1] <= $signed(muls[4]) + $signed(muls[5]) + $signed(muls[6]) + $signed(muls[7]);
            acc[2] <= $signed(muls[8]) + $signed(muls[9]) + $signed(muls[10]) + $signed(muls[11]);
            acc[3] <= $signed(muls[12]) + $signed(muls[13]) + $signed(muls[14]) + $signed(muls[15]);
            // normalize from Q32.32 (product) back to Q16.16 by shifting right 16 bits
            out_vec[0] <= acc[0][(16+31):16]; // extract 32-bit signed result
            out_vec[1] <= acc[1][(16+31):16];
            out_vec[2] <= acc[2][(16+31):16];
            out_vec[3] <= acc[3][(16+31):16];
        end
    end
endmodule