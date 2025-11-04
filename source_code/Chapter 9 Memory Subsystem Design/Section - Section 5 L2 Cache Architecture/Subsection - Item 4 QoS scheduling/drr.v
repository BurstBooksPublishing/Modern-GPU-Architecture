module drr_qos #(
  parameter C = 4,                     // number of classes
  parameter LEN_W = 12,                // length width (beats)
  parameter QUNIT_W = 12               // quantum unit width
)(
  input  wire                  clk,
  input  wire                  reset_n,
  input  wire [C-1:0]          req_valid,         // class head valid
  input  wire [C*LEN_W-1:0]    req_len_flat,      // concatenated lens MSB..LSB
  input  wire [C*QUNIT_W-1:0]  weight_flat,       // per-class weight
  output reg  [C-1:0]          grant,             // one-hot grant
  output reg                   grant_valid
);
  // unpack arrays
  reg [LEN_W-1:0] req_len [0:C-1];
  reg [QUNIT_W-1:0] weight [0:C-1];
  integer i;
  always @(*) begin
    for (i=0;i= req_len[idx]) begin
          deficit[idx] <= deficit[idx] - req_len[idx];
          grant[idx] <= 1'b1; grant_valid <= 1'b1;
          ptr <= (idx + 1) % C;
          starvation_cnt[idx] <= 0;
          disable for;
        end else begin
          starvation_cnt[idx] <= starvation_cnt[idx] + 1;
          // optional aging: boost deficit if starved
          if (starvation_cnt[idx] == 16'hFFFF) deficit[idx] <= deficit[idx] + {{(QUNIT_W+LEN_W+1-LEN_W){1'b0}}, req_len[idx]};
        end
      end
    end
  end
endmodule