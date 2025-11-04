module warp_mask_pipeline
  #(parameter WARP_SIZE = 32, parameter LANE_ID_WIDTH = 5, parameter CNT_WIDTH = 6)
  (input  wire                    clk,
   input  wire                    rst_n,
   input  wire [WARP_SIZE-1:0]    mask_in,     // incoming active mask
   input  wire [LANE_ID_WIDTH-1:0] lane_id,    // lane index
   output wire                    lane_active, // continuous combinational
   output reg  [WARP_SIZE-1:0]    mask_reg,    // registered pipeline stage
   output reg  [CNT_WIDTH-1:0]    active_count // combinational->registered
  );

  // continuous assignment: simple combinational view of a dynamic bit
  assign lane_active = mask_in[lane_id]; // stateless, optimized by synthesizer

  // combinational popcount implemented procedurally for synthesis
  integer i;
  reg [CNT_WIDTH-1:0] popcnt_comb;
  always @* begin
    popcnt_comb = {CNT_WIDTH{1'b0}};
    for (i = 0; i < WARP_SIZE; i = i + 1)
      popcnt_comb = popcnt_comb + mask_in[i];
  end

  // sequential pipeline: latch mask and popcount on clock edge (procedural)
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mask_reg     <= {WARP_SIZE{1'b0}};
      active_count <= {CNT_WIDTH{1'b0}};
    end else begin
      mask_reg     <= mask_in;       // stateful register (procedural)
      active_count <= popcnt_comb;   // register the combinational result
    end
  end

endmodule