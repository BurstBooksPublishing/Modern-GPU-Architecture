module gbuffer_tb #(
  parameter CLK_DIV = 2,           // clock divider
  parameter PIXELS_PER_FRAME = 65536,
  parameter LAYERS = 3
)(
  input  wire clk_in,
  input  wire rst_n_in
);
  // clock and reset (synchronous); clk derived for DUT if needed
  reg clk = 0;
  reg rst_n = 0;
  always @(posedge clk_in) begin
    clk <= ~clk;                      // simple clock toggle for test harness
    rst_n <= rst_n_in;
  end

  // Producer interface signals (synthesizable stimulus)
  reg [$clog2(PIXELS_PER_FRAME)-1:0] prod_cnt = 0;
  reg prod_valid = 0;
  reg [7:0] prod_vrs_mask = 8'hFF;    // coarse shading mask
  reg [31:0] prod_addr = 0;
  wire prod_ready;

  // Simple FSM to produce a burst then stall (backpressure scenario)
  reg [1:0] state = 0;
  always @(posedge clk) begin
    if (!rst_n) begin
      state <= 0; prod_cnt <= 0; prod_valid <= 0; prod_addr <= 0;
    end else begin
      case(state)
        0: begin prod_valid <= 1; prod_cnt <= 0; state <= 1; end
        1: begin
             if (prod_ready) begin
               prod_addr <= prod_addr + 1; prod_cnt <= prod_cnt + 1;
               if (prod_cnt == PIXELS_PER_FRAME-1) begin prod_valid <= 0; state <= 2; end
             end
           end
        2: begin // idle to allow drain
             if (prod_ready) state <= 0;
           end
      endcase
    end
  end

  // DUT instantiation (assumed synthesizable module)
  // Interface signals chosen to be common in G-buffer managers.
  wire dbg_full;
  gbuffer_mgr dut (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(prod_valid),
    .in_ready(prod_ready),
    .in_addr(prod_addr),
    .in_vrs_mask(prod_vrs_mask),
    .dbg_full(dbg_full)
    // additional ports omitted for brevity in harness example
  );

  // Simple coverage counters (synthesizable)
  reg [31:0] writes = 0;
  always @(posedge clk) if (rst_n && prod_valid && prod_ready) writes <= writes + 1;

endmodule