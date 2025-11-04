module compression_mode_arbiter(
    input  wire        clk,         // system clock
    input  wire        rst_n,       // active-low reset
    input  wire [1:0]  ratio_hint,  // 00=low,01=mid,10=high target ratio
    input  wire [7:0]  latency_budget_cycles, // allowed decode latency
    output reg         use_high_quality, // 1=>choose low-R high-quality decoder
    output reg         prefetch_enable    // enable cache prefetching for heavy decode
);
    // parameters derived from microbenchmarking per-format
    localparam HIGHQ_LATENCY = 16; // cycles needed for high-quality decoder
    localparam MIDQ_LATENCY  = 8;
    localparam LOWQ_LATENCY  = 4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            use_high_quality <= 1'b0;
            prefetch_enable  <= 1'b0;
        end else begin
            case (ratio_hint)
                2'b10: begin // high ratio target -> try low-cost decode
                    use_high_quality <= (latency_budget_cycles >= HIGHQ_LATENCY) ? 1'b1 : 1'b0;
                end
                2'b01: begin // mid
                    use_high_quality <= (latency_budget_cycles >= MIDQ_LATENCY) ? 1'b1 : 1'b0;
                end
                default: begin // low ratio -> prefer quality
                    use_high_quality <= 1'b1;
                end
            endcase
            // simple hysteresis: enable prefetch if high throughput and high ratio
            prefetch_enable <= (ratio_hint == 2'b10) && (latency_budget_cycles > LOWQ_LATENCY);
        end
    end
endmodule