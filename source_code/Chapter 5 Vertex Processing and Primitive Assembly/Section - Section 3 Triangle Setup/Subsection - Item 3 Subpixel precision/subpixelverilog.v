module subpixel_edge_setup #(
    parameter W = 32,           // total input width
    parameter FRAC = 4          // fractional bits (1/16 default)
)(
    input  signed [W-1:0] x0, y0, // vertex 0 (fixed-point)
    input  signed [W-1:0] x1, y1, // vertex 1
    input  signed [W-1:0] x2, y2, // vertex 2 (unused by single-edge example)
    input  signed [W-1:0] px, py, // pixel integer coordinates (fixed-point aligned)
    output signed [W+W:0] edge_val // edge function result (wider to hold mult)
);
    // Edge coefficients (no additional scaling beyond vertex fixed point)
    wire signed [W-1:0] A = y1 - y0;                     // A = y1 - y0
    wire signed [W-1:0] B = -(x1 - x0);                  // B = -(x1 - x0)
    // C = x1*y0 - y1*x0  (needs double width)
    wire signed [2*W-1:0] C = (x1 * y0) - (y1 * x0);

    // Add subpixel offset 0.5 -> represented as 1 << (FRAC-1)
    localparam signed [W-1:0] HALF = (1 << (FRAC-1));
    wire signed [W-1:0] sx = px + HALF;
    wire signed [W-1:0] sy = py + HALF;

    // Multiply A*sx and B*sy with extended widths
    wire signed [2*W-1:0] Asx = A * sx;
    wire signed [2*W-1:0] Bsy = B * sy;

    // Combine; result is in same fixed-point scale as C (vertex scale squared handled)
    assign edge_val = Asx + Bsy + C;
endmodule