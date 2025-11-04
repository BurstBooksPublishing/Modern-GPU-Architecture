module scanout_controller #(
  parameter ADDR_W = 32,
  parameter STRIDE_W = 32,
  parameter X_W = 12,            // up to 4096 pixels per line
  parameter Y_W = 12,
  parameter PIXEL_BYTES = 4      // RGBA8 default
) (
  input  wire                 pix_clk, rst_n,
  input  wire                 enable,                // start/stop scanout
  input  wire [ADDR_W-1:0]    frame_base,            // base address of current buffer
  input  wire [STRIDE_W-1:0]  stride,                // bytes per line
  input  wire [X_W-1:0]       width,                 // pixels per line
  input  wire [Y_W-1:0]       height,
  // memory read request interface
  output reg                  mem_req,
  output reg [ADDR_W-1:0]     mem_addr,
  input  wire                 mem_ready,
  input  wire [PIXEL_BYTES*8-1:0] mem_rdata,
  input  wire                 mem_rvalid,
  // pixel output
  output reg [PIXEL_BYTES*8-1:0] pixel_out,
  output reg                  pixel_valid,
  // sync (optional, driven from timing generator)
  input  wire                 hsync, vsync
);

  // counters
  reg [X_W-1:0] x_cnt;
  reg [Y_W-1:0] y_cnt;

  // simple one-cycle request pipeline: request for next pixel
  // request lead = 1 cycle (adjust for real memory latency)
  wire [ADDR_W-1:0] next_addr = frame_base + (y_cnt * stride)
                                 + (x_cnt * PIXEL_BYTES);

  always @(posedge pix_clk or negedge rst_n) begin
    if (!rst_n) begin
      x_cnt <= 0; y_cnt <= 0;
      mem_req <= 0; mem_addr <= 0;
      pixel_out <= 0; pixel_valid <= 0;
    end else if (enable) begin
      if (hsync) begin
        x_cnt <= 0;                         // start of line
      end
      if (vsync) begin
        y_cnt <= 0;                         // start of frame
      end
      // Issue mem request when previous accepted or idle
      if (!mem_req && mem_ready) begin
        mem_req <= 1;
        mem_addr <= next_addr;             // compute address per eq. (1)
      end else if (mem_req && mem_ready) begin
        mem_req <= 0;                       // request accepted
      end
      // consume returned pixel
      if (mem_rvalid) begin
        pixel_out <= mem_rdata;
        pixel_valid <= 1;
        // advance pixel counters
        if (x_cnt + 1 == width) begin
          x_cnt <= 0;
          if (y_cnt + 1 == height) y_cnt <= 0; else y_cnt <= y_cnt + 1;
        end else x_cnt <= x_cnt + 1;
      end else begin
        pixel_valid <= 0;
      end
    end else begin
      mem_req <= 0; pixel_valid <= 0;
    end
  end

endmodule