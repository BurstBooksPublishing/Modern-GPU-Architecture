module lane_aggregator #(
  parameter NUM_LANES = 4,
  parameter LANE_WIDTH = 32
)(
  input  wire                    clk,
  input  wire                    rst_n,
  input  wire [NUM_LANES-1:0]    lane_valid,   // per-lane-valid
  input  wire [NUM_LANES*LANE_WIDTH-1:0] lane_data, // concatenated inputs
  output reg                     out_valid,
  output reg [NUM_LANES*LANE_WIDTH-1:0] out_data,
  input  wire                    out_ready
);
  // latch when all lanes present
  wire all_valid = &lane_valid;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
      out_data  <= {NUM_LANES*LANE_WIDTH{1'b0}};
    end else begin
      if (all_valid && !out_valid) begin
        out_data  <= lane_data; // aggregate atomically
        out_valid <= 1'b1;
      end else if (out_valid && out_ready) begin
        out_valid <= 1'b0;      // handshake completed
      end
    end
  end
endmodule