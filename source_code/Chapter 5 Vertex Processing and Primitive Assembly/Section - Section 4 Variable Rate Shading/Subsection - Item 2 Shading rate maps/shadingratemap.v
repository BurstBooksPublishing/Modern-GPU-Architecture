module shading_rate_map #(
  parameter MAP_W = 64, // map width in tiles
  parameter MAP_H = 36, // map height in tiles
  parameter TILE_W_LOG2 = 1, // log2(tile width)
  parameter TILE_H_LOG2 = 1  // log2(tile height)
) (
  input  wire                clk,
  input  wire                rst_n,
  input  wire [15:0]         pix_x, // pixel x coordinate
  input  wire [15:0]         pix_y, // pixel y coordinate
  input  wire                req,   // request single-shot
  output reg  [1:0]          rx,    // shading rate x (1->1,2->2,4->4 encoded)
  output reg  [1:0]          ry,    // shading rate y
  output reg                 valid
);

  // compute tile indices by shifting when TILE_* are power-of-two
  wire [15:0] tile_x = pix_x >> TILE_W_LOG2;
  wire [15:0] tile_y = pix_y >> TILE_H_LOG2;
  wire [15:0] idx = tile_y * MAP_W + tile_x;

  // simple single-ported ROM implemented as distributed RAM (init externally)
  // width 4 bits: [rx(2b)|ry(2b)]
  reg [3:0] map_mem [0:MAP_W*MAP_H-1];

  // synchronous read: one-cycle latency
  reg read_en;
  reg [15:0] read_addr;
  always @(posedge clk) begin
    if (!rst_n) begin
      read_en   <= 1'b0;
      read_addr <= 0;
      valid     <= 1'b0;
    end else begin
      if (req) begin
        read_en   <= 1'b1;
        read_addr <= idx;
      end else begin
        read_en <= 1'b0;
      end
      if (read_en) begin
        {rx,ry} <= map_mem[read_addr]; // deliver encoded rates
        valid   <= 1'b1;
      end else begin
        valid <= 1'b0;
      end
    end
  end

endmodule