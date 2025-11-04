module display_timing_gen #(
  parameter integer H_ACTIVE = 1920,
  parameter integer H_FP     = 88,
  parameter integer H_SYNC   = 44,
  parameter integer H_BP     = 148,
  parameter integer V_ACTIVE = 1080,
  parameter integer V_FP     = 4,
  parameter integer V_SYNC   = 5,
  parameter integer V_BP     = 36,
  parameter        HSYNC_POL = 1'b0, // 0: active low, 1: active high
  parameter        VSYNC_POL = 1'b0
)(
  input  wire clk,           // pixel clock
  input  wire rst_n,         // active-low sync reset
  output reg  hsync,
  output reg  vsync,
  output reg  de,            // data enable
  output reg  [15:0] pixel_x,// current pixel within active (0..H_ACTIVE-1)
  output reg  [15:0] pixel_y // current line within active (0..V_ACTIVE-1)
);
  // compute totals
  localparam integer H_TOTAL = H_ACTIVE + H_FP + H_SYNC + H_BP;
  localparam integer V_TOTAL = V_ACTIVE + V_FP + V_SYNC + V_BP;

  reg [31:0] h_count;
  reg [31:0] v_count;

  // synchronous counters and signal generation
  always @(posedge clk) begin
    if (!rst_n) begin
      h_count <= 0;
      v_count <= 0;
      hsync   <= ~HSYNC_POL;
      vsync   <= ~VSYNC_POL;
      de      <= 0;
      pixel_x <= 0;
      pixel_y <= 0;
    end else begin
      // horizontal counter
      if (h_count == H_TOTAL-1) begin
        h_count <= 0;
        // vertical increment on horizontal wrap
        if (v_count == V_TOTAL-1)
          v_count <= 0;
        else
          v_count <= v_count + 1;
      end else begin
        h_count <= h_count + 1;
      end

      // HSYNC active during sync window (after front porch)
      if ((h_count >= H_ACTIVE + H_FP) && (h_count < H_ACTIVE + H_FP + H_SYNC))
        hsync <= HSYNC_POL;
      else
        hsync <= ~HSYNC_POL;

      // VSYNC active during vertical sync window
      if ((v_count >= V_ACTIVE + V_FP) && (v_count < V_ACTIVE + V_FP + V_SYNC))
        vsync <= VSYNC_POL;
      else
        vsync <= ~VSYNC_POL;

      // Data enable when within active region
      if ((h_count < H_ACTIVE) && (v_count < V_ACTIVE)) begin
        de <= 1'b1;
        pixel_x <= h_count[15:0];
        pixel_y <= v_count[15:0];
      end else begin
        de <= 1'b0;
        pixel_x <= 0;
        pixel_y <= 0;
      end
    end
  end
endmodule