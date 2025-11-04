module barrier_sync #(
    parameter N_PARTICIPANTS = 32,
    parameter GEN_WIDTH = 8
) (
    input  wire                     clk,
    input  wire                     rst_n,       // active low reset
    input  wire                     arrive,      // single-bit per-cycle arrival strobe
    input  wire [$clog2(N_PARTICIPANTS)-1:0] id,   // participant id
    output reg  [GEN_WIDTH-1:0]     gen_out,     // generation broadcast
    output reg                      barrier_pulse // one-cycle pulse when barrier completes
);
    localparam ID_WIDTH = $clog2(N_PARTICIPANTS);
    localparam CNT_WIDTH = $clog2(N_PARTICIPANTS+1);

    reg [GEN_WIDTH-1:0] gen;
    reg [N_PARTICIPANTS-1:0] arrived;
    reg [CNT_WIDTH-1:0] count;

    integer i;
    // reset logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gen <= {GEN_WIDTH{1'b0}};
            arrived <= {N_PARTICIPANTS{1'b0}};
            count <= {CNT_WIDTH{1'b0}};
            gen_out <= {GEN_WIDTH{1'b0}};
            barrier_pulse <= 1'b0;
        end else begin
            barrier_pulse <= 1'b0; // default clear
            if (arrive) begin
                // only count first arrival per participant for current generation
                if (!arrived[id]) begin
                    arrived[id] <= 1'b1;
                    count <= count + 1'b1;
                end
            end
            // check completion
            if (count == N_PARTICIPANTS - 1 && (arrive && !arrived[id] || count == N_PARTICIPANTS - 1)) begin
                // barrier reached: increment generation, clear arrived, reset count
                gen <= gen + 1'b1;
                gen_out <= gen + 1'b1; // broadcast next gen as release indicator
                arrived <= {N_PARTICIPANTS{1'b0}};
                count <= {CNT_WIDTH{1'b0}};
                barrier_pulse <= 1'b1; // one-cycle global pulse
            end
        end
    end
endmodule