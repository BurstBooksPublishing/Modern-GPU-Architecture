module coarse_shade_mapper #(
    parameter TILE_W = 16,
    parameter TILE_H = 16,
    parameter COORD_W = 8
)(
    input  wire [COORD_W-1:0] pixel_x, // pixel x in tile
    input  wire [COORD_W-1:0] pixel_y, // pixel y in tile
    input  wire [1:0] rate_log2,       // 0:1x1,1:2x2,2:4x4
    output wire [COORD_W-1:0] cen_x,   // centroid x for shading lookup
    output wire [COORD_W-1:0] cen_y,   // centroid y
    output wire [COORD_W-1:0] group_id // linear group id within tile
);
    // compute group indices via shift
    wire [COORD_W-1:0] gx = pixel_x >> rate_log2;
    wire [COORD_W-1:0] gy = pixel_y >> rate_log2;
    // centroid offset = half the region size
    wire [COORD_W-1:0] half = 1 << (rate_log2 - 1); // safe when rate_log2>0
    // centroid coordinates (clamp within tile)
    assign cen_x = (rate_log2==0) ? pixel_x : ((gx << rate_log2) + half);
    assign cen_y = (rate_log2==0) ? pixel_y : ((gy << rate_log2) + half);
    // linear group id = gy * (TILE_W >> rate_log2) + gx
    wire [COORD_W-1:0] groups_per_row = TILE_W >> rate_log2;
    assign group_id = gy * groups_per_row + gx;
endmodule