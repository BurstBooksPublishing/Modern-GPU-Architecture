module bc1_decode(
    input  wire [63:0] block_in,      // [63:48]=c1, [47:32]=c0, [31:0]=indices (2b per pixel)
    output wire [511:0] pixels_out    // 16 pixels * 32-bit RGBA (pixels_out[31:0] = pixel0)
);
    // extract fields
    wire [15:0] c1 = block_in[63:48];
    wire [15:0] c0 = block_in[47:32];
    wire [31:0] idx = block_in[31:0];

    // expand RGB565 to 8-bit channels
    function [23:0] expand565(input [15:0] c);
        reg [4:0] r5;
        reg [5:0] g6;
        reg [4:0] b5;
        reg [7:0] r8, g8, b8;
    begin
        r5 = c[15:11];
        g6 = c[10:5];
        b5 = c[4:0];
        r8 = {r5, r5[4:2]};        // replicate high bits
        g8 = {g6, g6[5:4]};
        b8 = {b5, b5[4:2]};
        expand565 = {r8,g8,b8};
    end
    endfunction

    wire [23:0] e0 = expand565(c0);
    wire [23:0] e1 = expand565(c1);

    // compute palette entries
    reg [23:0] pal [0:3];
    integer i;
    always @(*) begin
        pal[0] = e0;
        pal[1] = e1;
        if (c0 > c1) begin
            // 4-color mode: interpolate (per-channel)
            pal[2][23:16] = ( (2*e0[23:16] + e1[23:16]) / 3 );
            pal[2][15:8]  = ( (2*e0[15:8]  + e1[15:8])  / 3 );
            pal[2][7:0]   = ( (2*e0[7:0]   + e1[7:0])   / 3 );
            pal[3][23:16] = ( (e0[23:16] + 2*e1[23:16]) / 3 );
            pal[3][15:8]  = ( (e0[15:8]  + 2*e1[15:8])  / 3 );
            pal[3][7:0]   = ( (e0[7:0]   + 2*e1[7:0])   / 3 );
        end else begin
            // 3-color + transparent
            pal[2][23:0] = ( (e0 + e1) >> 1 ); // average
            pal[3] = 24'h000000; // treated as transparent
        end
    end

    // assemble output pixels
    reg [31:0] pixels [0:15];
    always @(*) begin
        for (i=0;i<16;i=i+1) begin
            reg [1:0] sel;
            sel = idx >> (2*i);
            sel = sel & 2'b11;
            if ((c0 <= c1) && (sel == 2'b11)) begin
                pixels[i] = {8'h00, 24'h000000}; // alpha=0 -> transparent
            end else begin
                pixels[i] = {8'hFF, pal[sel]}; // alpha=255
            end
        end
    end

    // flatten output
    genvar g;
    generate
      for (g=0; g<16; g=g+1) begin : pack
        assign pixels_out[32*g +: 32] = pixels[g];
      end
    endgenerate
endmodule