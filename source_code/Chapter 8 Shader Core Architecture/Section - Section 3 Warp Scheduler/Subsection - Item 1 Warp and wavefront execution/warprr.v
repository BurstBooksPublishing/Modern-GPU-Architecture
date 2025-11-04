module warp_rr #(
  parameter W = 16,               // number of warps
  parameter ID_BITS = $clog2(W)
) (
  input  wire                 clk,
  input  wire                 rst,
  input  wire [W-1:0]         ready_vec,   // per-warp ready
  output reg  [ID_BITS-1:0]   grant_id,    // selected warp id
  output reg                  grant_valid  // high when a grant exists
);
  reg [W-1:0] ptr;                  // round-robin pointer
  integer i;
  always @(posedge clk) begin
    if (rst) begin
      ptr <= {{(W-1){1'b0}},1'b1};  // pointer at warp 0
      grant_valid <= 1'b0;
      grant_id <= {ID_BITS{1'b0}};
    end else begin
      grant_valid <= 1'b0;
      // search W entries starting at pointer
      for (i=0;i> (W-1)); // rotate pointer
          disable for;
        end
      end
    end
  end
endmodule