module stage_dispatcher #(
  parameter NUM_SM = 8,              // number of SMs
  parameter ID_WIDTH = 32,
  parameter SM_IDX_W = $clog2(NUM_SM)
)(
  input  wire                  clk,
  input  wire                  rstn,
  input  wire [2:0]            stage_type, // 0=vertex,1=frag,2=compute,3=ray...
  input  wire [ID_WIDTH-1:0]   job_id,
  input  wire                  in_valid,
  output reg                   in_ready,
  output reg  [SM_IDX_W-1:0]   out_sm,    // selected SM index
  output reg                   out_valid,
  input  wire                  out_ready
);
  // simple round-robin pointer
  reg [SM_IDX_W-1:0] rr_ptr;
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      rr_ptr <= 0;
      in_ready <= 1'b1;
      out_valid <= 1'b0;
    end else begin
      if (in_valid && in_ready) begin
        out_sm <= rr_ptr;
        out_valid <= 1'b1;
        in_ready <= 1'b0; // wait for downstream accept
        rr_ptr <= rr_ptr + 1'b1;
      end else if (out_valid && out_ready) begin
        out_valid <= 1'b0;
        in_ready <= 1'b1;
      end
    end
  end
endmodule