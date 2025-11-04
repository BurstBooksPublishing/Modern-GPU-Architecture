module noc_vc_manager #(
  parameter NUM_VC = 8,
  parameter C_WIDTH = 8
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire [NUM_VC-1:0]    alloc_req,      // request per VC class (one-hot semantics)
  output reg  [NUM_VC-1:0]    alloc_grant,    // grant per VC (one-hot)
  input  wire [$clog2(NUM_VC)-1:0] free_idx,  // index returned when flit leaves downstream
  input  wire                 free_valid,     // free strobe increments credit
  input  wire [C_WIDTH-1:0]   init_credits    // initial credit value load
);
  // per-VC credit counters
  reg [C_WIDTH-1:0] credits [0:NUM_VC-1];
  integer i;
  // initialize credits
  always @(posedge clk) begin
    if (rst) begin
      for (i=0;i0 wins; escape VC = 0
      alloc_grant <= {NUM_VC{1'b0}};
      for (i=0;i0 && alloc_grant=={NUM_VC{1'b0}}) begin
          alloc_grant[i] <= 1'b1;
          credits[i] <= credits[i] - 1'b1; // consume credit on allocation
        end
      end
      // if no regular grant, try escape VC0 if requested and has credit
      if (alloc_grant=={NUM_VC{1'b0}} && alloc_req[0] && credits[0]>0) begin
        alloc_grant[0] <= 1'b1;
        credits[0] <= credits[0] - 1'b1;
      end
    end
  end
endmodule