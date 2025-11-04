module pipelined_mat4x4 (
    input  wire         clk,      // system clock
    input  wire         rst_n,    // active-low reset
    input  wire         in_valid, // input valid
    input  wire signed [31:0] mat [0:15], // 4x4 matrix, row-major
    input  wire signed [31:0] vec [0:3],  // input vector
    output reg  signed [63:0] out [0:3],  // output vector (wider to hold sums)
    output reg          out_valid // output valid
);
    // Stage 0: compute two rows (row0,row1) partially and register results.
    reg signed [63:0] stage0_y0, stage0_y1;
    reg               stage0_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage0_y0   <= 64'sd0;
            stage0_y1   <= 64'sd0;
            stage0_valid<= 1'b0;
        end else begin
            // combinational MACs grouped: two rows per stage
            stage0_y0 <= mat[0]*vec[0] + mat[1]*vec[1] + mat[2]*vec[2] + mat[3]*vec[3];
            stage0_y1 <= mat[4]*vec[0] + mat[5]*vec[1] + mat[6]*vec[2] + mat[7]*vec[3];
            stage0_valid <= in_valid;
        end
    end

    // Stage 1: compute remaining rows and produce final outputs.
    reg signed [63:0] stage1_y2, stage1_y3;
    reg               stage1_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_y2   <= 64'sd0;
            stage1_y3   <= 64'sd0;
            stage1_valid<= 1'b0;
            out_valid   <= 1'b0;
            out[0] <= 64'sd0; out[1] <= 64'sd0; out[2] <= 64'sd0; out[3] <= 64'sd0;
        end else begin
            // compute row2,row3 combinationally
            stage1_y2 <= mat[8]*vec[0]  + mat[9]*vec[1] + mat[10]*vec[2] + mat[11]*vec[3];
            stage1_y3 <= mat[12]*vec[0] + mat[13]*vec[1]+ mat[14]*vec[2] + mat[15]*vec[3];
            stage1_valid <= stage0_valid;

            // final outputs produced next cycle: combine registered stage0 with stage1 results
            out[0] <= stage0_y0; // already full row sum
            out[1] <= stage0_y1;
            out[2] <= stage1_y2;
            out[3] <= stage1_y3;
            out_valid <= stage1_valid;
        end
    end
endmodule