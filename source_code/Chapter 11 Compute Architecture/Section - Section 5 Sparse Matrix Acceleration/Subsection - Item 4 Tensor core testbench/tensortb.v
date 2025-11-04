`timescale 1ns/1ps
module tensor_core_2x2 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    output wire        in_ready,
    input  wire signed [15:0] a00,a01,a10,a11, // FP16 encoded as 16-bit words
    input  wire signed [15:0] b00,b01,b10,b11,
    output reg         out_valid,
    input  wire        out_ready,
    output reg signed [31:0] c00,c01,c10,c11   // 32-bit accumulator
);
    // Simple pipeline stage: latch inputs, compute integer multiply-accumulate
    reg        stage_valid;
    reg signed [15:0] ra00,ra01,ra10,ra11;
    reg signed [15:0] rb00,rb01,rb10,rb11;
    assign in_ready = !stage_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage_valid <= 1'b0;
            out_valid <= 1'b0;
            c00 <= 0; c01 <= 0; c10 <= 0; c11 <= 0;
        end else begin
            // input latch when ready
            if (in_valid && in_ready) begin
                stage_valid <= 1'b1;
                ra00<=a00; ra01<=a01; ra10<=a10; ra11<=a11;
                rb00<=b00; rb01<=b01; rb10<=b10; rb11<=b11;
            end
            // computation completes next cycle
            if (stage_valid) begin
                // integer multiply accumulate as proxy for FP fused op
                c00 <= c00 + ra00*rb00 + ra01*rb10;
                c01 <= c01 + ra00*rb01 + ra01*rb11;
                c10 <= c10 + ra10*rb00 + ra11*rb10;
                c11 <= c11 + ra10*rb01 + ra11*rb11;
                out_valid <= 1'b1;
                stage_valid <= 1'b0;
            end else if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end
        end
    end
endmodule

module tb_tensor_core;
    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk; // 100MHz clock for simulation

    // DUT signals
    reg in_valid;
    wire in_ready;
    reg signed [15:0] a00,a01,a10,a11,b00,b01,b10,b11;
    wire out_valid;
    reg out_ready;
    wire signed [31:0] c00,c01,c10,c11;

    tensor_core_2x2 DUT(
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_ready(in_ready),
        .a00(a00),.a01(a01),.a10(a10),.a11(a11),
        .b00(b00),.b01(b01),.b10(b10),.b11(b11),
        .out_valid(out_valid), .out_ready(out_ready),
        .c00(c00),.c01(c01),.c10(c10),.c11(c11)
    );

    integer cycle_count = 0;
    integer errors = 0;
    integer completed = 0;
    // golden reference accumulators
    integer g00=0,g01=0,g10=0,g11=0;

    // deterministic stimulus FSM
    initial begin
        rst_n = 0; in_valid = 0; out_ready = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        // feed 16 tiles with random stall injection
        repeat (16) begin
            // generate small integer operands to avoid overflow when multiplied
            a00 = $urandom_range(-8,8); a01 = $urandom_range(-8,8);
            a10 = $urandom_range(-8,8); a11 = $urandom_range(-8,8);
            b00 = $urandom_range(-8,8); b01 = $urandom_range(-8,8);
            b10 = $urandom_range(-8,8); b11 = $urandom_range(-8,8);
            // push tile when DUT ready
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            in_valid = 1;
            @(posedge clk);
            in_valid = 0;
            // update golden reference immediately for this simple pipeline
            g00 = g00 + a00*b00 + a01*b10;
            g01 = g01 + a00*b01 + a01*b11;
            g10 = g10 + a10*b00 + a11*b10;
            g11 = g11 + a10*b01 + a11*b11;
            // randomly allow backpressure on the read side
            out_ready = $urandom_range(0,1);
            // allow some cycles between tiles
            repeat ($urandom_range(0,2)) @(posedge clk);
        end
        // wait for remaining outputs
        repeat (10) @(posedge clk);
        $display("Completed: %0d, Errors: %0d, Cycles: %0d", completed, errors, cycle_count);
        $finish;
    end

    // monitor outputs and compare to golden
    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (out_valid && out_ready) begin
            completed = completed + 1;
            if (c00 !== g00) errors = errors + 1;
            if (c01 !== g01) errors = errors + 1;
            if (c10 !== g10) errors = errors + 1;
            if (c11 !== g11) errors = errors + 1;
        end
    end
endmodule