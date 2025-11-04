`timescale 1ns/1ps
module stream_gen #(
  parameter H_PIX = 1280, V_LINES = 720
)(
  input wire clk, input wire rst_n,
  output reg [23:0] pixel, output reg valid, output reg last
);
  reg [31:0] x, y;
  always @(posedge clk) begin
    if (!rst_n) begin x <= 0; y <= 0; valid <= 0; last <= 0; pixel <= 0; end
    else begin
      valid <= 1;
      pixel <= {8'd0, x[7:0], y[7:0]}; // deterministic pattern
      last <= (x == H_PIX-1) && (y == V_LINES-1);
      if (x == H_PIX-1) begin x <= 0; if (y == V_LINES-1) y <= 0; else y <= y+1; end
      else x <= x+1;
    end
  end
endmodule

module video_pipeline #(
  parameter LATENCY = 4
)(
  input  wire clk, input wire rst_n,
  input  wire [23:0] in_pixel, input wire in_valid, output reg in_ready,
  output reg [23:0] out_pixel, output reg out_valid, input wire out_ready
);
  // simple shift-register latency pipeline with valid-ready
  reg [23:0] pipe_pix [0:LATENCY-1];
  reg        pipe_v   [0:LATENCY-1];
  integer i;
  always @(posedge clk) begin
    if (!rst_n) begin
      for (i=0;i0;i=i-1) begin
        if (~pipe_v[i] && pipe_v[i-1]) begin pipe_pix[i] <= pipe_pix[i-1]; pipe_v[i] <= 1; pipe_v[i-1] <= 0; end
      end
      // output stage
      if (pipe_v[LATENCY-1]) begin
        if (out_ready) begin out_pixel <= pipe_pix[LATENCY-1]; out_valid <= 1; pipe_v[LATENCY-1] <= 0; end
        else out_valid <= 1; // backpressure holds valid high
      end else out_valid <= 0;
    end
  end
endmodule

module stream_check(
  input wire clk, input wire rst_n,
  input wire [23:0] pixel, input wire valid, input wire last
);
  reg [31:0] seen;
  always @(posedge clk) begin
    if (!rst_n) begin seen <= 0; end
    else if (valid) begin
      seen <= seen + 1;
      if (last) $display("Frame completed: %0d pixels seen", seen);
    end
  end
endmodule

module tb_top;
  reg clk = 0; always #5 clk = ~clk;
  reg rst_n = 0;
  wire [23:0] pix_a; wire v_a, l_a;
  wire [23:0] pix_b; wire v_b, r_b;
  stream_gen #(.H_PIX(640), .V_LINES(480)) gen(.clk(clk), .rst_n(rst_n), .pixel(pix_a), .valid(v_a), .last(l_a));
  video_pipeline #(.LATENCY(8)) dut(.clk(clk), .rst_n(rst_n), .in_pixel(pix_a), .in_valid(v_a), .in_ready(r_b), .out_pixel(pix_b), .out_valid(v_b), .out_ready(1'b1));
  stream_check chk(.clk(clk), .rst_n(rst_n), .pixel(pix_b), .valid(v_b), .last(l_a));
  initial begin
    #20 rst_n = 1;
    #200000 $finish; // limit simulation time
  end
endmodule