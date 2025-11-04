`timescale 1ns/1ps
module tb_texture_pipeline;
  parameter ADDR_W = 32, DATA_W = 128;
  reg clk = 0, rst_n = 0;
  // DUT stimulus: UV, mip, control
  reg        in_valid;
  reg [31:0] in_uv_u, in_uv_v;
  reg [3:0]  in_mip_level;
  wire       out_valid;
  wire [DATA_W-1:0] out_color;
  // Memory interface
  wire        mem_req;
  wire [ADDR_W-1:0] mem_addr;
  wire        mem_ack;
  wire [DATA_W-1:0] mem_data;
  // Clock
  always #5 clk = ~clk;
  initial begin
    rst_n = 0; in_valid = 0;
    #20 rst_n = 1;
    // coherent burst: adjacent UVs to stress cache lines
    repeat (64) begin
      @(posedge clk);
      in_valid <= 1; in_uv_u <= $urandom_range(0,1023);
      in_uv_v <= $urandom_range(0,1023); in_mip_level <= 0;
    end
    in_valid <= 0;
    #200;
    // random pattern: increase miss rate
    repeat (128) begin
      @(posedge clk);
      in_valid <= ($urandom_range(0,3)==0); // sparse random requests
      in_uv_u <= $urandom; in_uv_v <= $urandom; in_mip_level <= $urandom_range(0,4);
    end
    #500 $finish;
  end
  // Instantiate DUT (connect names to your RTL)
  texture_pipeline dut (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .in_uv_u(in_uv_u), .in_uv_v(in_uv_v), .in_mip(in_mip_level),
    .out_valid(out_valid), .out_color(out_color),
    .mem_req(mem_req), .mem_addr(mem_addr), .mem_ack(mem_ack), .mem_data(mem_data)
  );
  // Simple memory model: respond after LAT cycles, configurable
  parameter LAT = 40;
  reg mem_ack_reg = 0;
  reg [DATA_W-1:0] mem_data_reg = 0;
  assign mem_ack = mem_ack_reg;
  assign mem_data = mem_data_reg;
  always @(posedge clk) begin
    mem_ack_reg <= 0;
    if (mem_req) begin
      // simulate variable latency and data pattern (checkerboard)
      fork
        begin
          repeat (LAT) @(posedge clk);
          mem_data_reg <= {DATA_W{1'b0}} ^ mem_addr; // simple reproducible pattern
          mem_ack_reg <= 1;
        end
      join_none
    end
  end
  // Simple checker: validate some outputs when out_valid asserted
  always @(posedge clk) if (out_valid) $display("sample out at %0t color[15:0]=%h",$time,out_color[15:0]);
endmodule