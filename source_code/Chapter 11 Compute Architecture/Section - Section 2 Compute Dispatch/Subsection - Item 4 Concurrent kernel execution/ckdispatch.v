module concurrent_kernel_dispatch #(
  parameter NUM_SMS = 8,
  parameter MAX_CTXS = 16,
  parameter ID_WIDTH = 32,
  parameter WG_WIDTH = 16
)(
  input  wire                   clk,
  input  wire                   rst,
  // launch interface
  input  wire                   launch_valid,
  input  wire [ID_WIDTH-1:0]    launch_id,
  input  wire [WG_WIDTH-1:0]    launch_wg_total,
  input  wire [15:0]            launch_regs_per_wg,
  input  wire [15:0]            launch_smem_per_wg,
  output reg                    launch_accept,
  // SM status inputs (free_slot indicates can accept another WG)
  input  wire [NUM_SMS-1:0]     sm_free_slot,
  // dispatch outputs (one-hot per SM with context id)
  output reg [NUM_SMS-1:0]      dispatch_valid,
  output reg [NUM_SMS*ID_WIDTH-1:0] dispatch_ctx_id
);

  // Context table
  reg [ID_WIDTH-1:0] ctx_id   [0:MAX_CTXS-1];
  reg [WG_WIDTH-1:0] ctx_rem  [0:MAX_CTXS-1];
  reg [15:0]         ctx_regs [0:MAX_CTXS-1];
  reg [15:0]         ctx_smem [0:MAX_CTXS-1];
  reg                ctx_used [0:MAX_CTXS-1];

  integer i, s;
  // accept new launch if a free context slot exists
  always @(posedge clk) begin
    if (rst) begin
      launch_accept <= 0;
      for (i=0;i0) begin
              // allocate one WG to SM s
              ctx_rem[i] <= ctx_rem[i] - 1;
              dispatch_valid[s] <= 1;
              dispatch_ctx_id[s*ID_WIDTH +: ID_WIDTH] <= ctx_id[i];
              // free context when done
              if (ctx_rem[i]==1) ctx_used[i] <= 0;
              disable for; // stop scanning contexts for this SM
            end
          end
        end
      end
    end
  end

endmodule