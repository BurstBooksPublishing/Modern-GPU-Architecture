`timescale 1ns/1ps
module rop_tb;
  // Clock/reset
  reg clk = 0;
  reg rst_n = 0;
  always #5 clk = ~clk; // 100MHz

  // Parameters (match DUT)
  parameter W = 32, H = 32;
  // DUT interface signals (simplified)
  reg        frag_valid;
  reg [31:0] frag_x, frag_y;
  reg [31:0] frag_color;
  reg [15:0] frag_depth;
  reg [3:0]  frag_sample_mask;
  wire       write_ack;

  // Instantiate DUT (assumed module name \lstinline|ROP_top|)
  ROP_top #(.WIDTH(W), .HEIGHT(H)) dut (
    .clk(clk),
    .rst_n(rst_n),
    .frag_valid(frag_valid),
    .frag_x(frag_x),
    .frag_y(frag_y),
    .frag_color(frag_color),
    .frag_depth(frag_depth),
    .frag_sample_mask(frag_sample_mask),
    .write_ack(write_ack)
  );

  // Simple framebuffer model
  reg [31:0] fb [0:W*H-1];
  integer i;
  initial begin
    // reset
    rst_n = 0; frag_valid = 0;
    #20 rst_n = 1;
    // directed sequence: overlapping fragments same pixel different depths
    send_frag(10,10,32'hFF0000FF, 100, 4'b1111);
    send_frag(10,10,32'h00FF00FF,  50, 4'b1111);
    // randomized stress
    for (i=0;i<1000;i=i+1) begin
      send_frag($urandom_range(0,W-1), $urandom_range(0,H-1),
                $urandom, $urandom_range(0,65535), $urandom_range(1,15));
    end
    #1000 $finish;
  end

  // task to send a fragment and wait for ack
  task send_frag(input integer x, input integer y, input [31:0] col,
                 input [15:0] z, input [3:0] mask);
    begin
      frag_x = x; frag_y = y; frag_color = col; frag_depth = z;
      frag_sample_mask = mask;
      frag_valid = 1;
      @(posedge clk);
      // wait for write_ack from DUT
      wait(write_ack);
      frag_valid = 0;
      @(posedge clk);
    end
  endtask

  // Simple checker that samples DUT framebuffer interface (hooked internally)
  // (Assumes DUT exposes a readback port; production testbenches use DPI or file I/O.)
endmodule