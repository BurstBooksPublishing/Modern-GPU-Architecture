module dtg #(
    parameter H_ACTIVE = 1920,
    parameter H_FP     = 88,
    parameter H_SYNC   = 44,
    parameter H_BP     = 148,
    parameter V_ACTIVE = 1080,
    parameter V_FP     = 4,
    parameter V_SYNC   = 5,
    parameter V_BP     = 36,
    parameter HSYNC_POL = 1'b0,  // active-low
    parameter VSYNC_POL = 1'b0
)(
    input  wire        clk,      // pixel clock
    input  wire        rst_n,
    output reg         hsync,
    output reg         vsync,
    output reg         hblank,
    output reg         vblank,
    output wire        pixel_valid
);

    localparam H_TOTAL = H_ACTIVE + H_FP + H_SYNC + H_BP;
    localparam V_TOTAL = V_ACTIVE + V_FP + V_SYNC + V_BP;

    reg [$clog2(H_TOTAL)-1:0] x;
    reg [$clog2(V_TOTAL)-1:0] y;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x <= 0;
            y <= 0;
            hsync  <= ~HSYNC_POL;
            vsync  <= ~VSYNC_POL;
            hblank <= 1'b1;
            vblank <= 1'b1;
        end else begin
            // horizontal counter
            if (x == H_TOTAL - 1) begin
                x <= 0;
                if (y == V_TOTAL - 1)
                    y <= 0;
                else
                    y <= y + 1;
            end else begin
                x <= x + 1;
            end

            // timing signals
            hblank <= (x >= H_ACTIVE);
            vblank <= (y >= V_ACTIVE);
            // HSYNC pulse window after front porch
            hsync <= ((x >= (H_ACTIVE + H_FP)) && (x < (H_ACTIVE + H_FP + H_SYNC)))
                     ? HSYNC_POL
                     : ~HSYNC_POL;
            // VSYNC pulse window after vertical front porch
            vsync <= ((y >= (V_ACTIVE + V_FP)) && (y < (V_ACTIVE + V_FP + V_SYNC)))
                     ? VSYNC_POL
                     : ~VSYNC_POL;
        end
    end

    assign pixel_valid = (x < H_ACTIVE) && (y < V_ACTIVE);

endmodule