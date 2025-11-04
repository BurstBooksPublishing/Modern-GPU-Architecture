module ctrl_datapath_top (
  input  wire         clk,
  input  wire         rstn,
  // input commands
  input  wire         cmd_valid,
  output wire         cmd_ready,
  input  wire [7:0]   cmd_warp,  // warp id
  input  wire [3:0]   cmd_op,
  input  wire [31:0]  cmd_imm,
  // output results
  output wire         res_valid,
  input  wire         res_ready,
  output wire [7:0]   res_warp,
  output wire [31:0]  res_data
);
  // small FSM issues when datapath ready
  reg pending;
  assign cmd_ready = !pending;
  reg [7:0] warp_q; reg [3:0] op_q; reg [31:0] imm_q;
  always @(posedge clk) begin
    if (!rstn) pending <= 1'b0;
    else if (cmd_valid && cmd_ready) begin
      pending <= 1'b1; warp_q <= cmd_warp; op_q <= cmd_op; imm_q <= cmd_imm;
    end else if (res_valid && res_ready) pending <= 1'b0; // retire
  end

  // datapath input handshake
  wire dp_in_valid = pending;
  wire dp_in_ready; // from datapath
  // feed datapath when ready
  wire [7:0] dp_warp = warp_q;
  wire [3:0] dp_op   = op_q;
  wire [31:0] dp_imm = imm_q;

  // instantiate datapath
  datapath #(.LATENCY(3)) U_DP (
    .clk(clk), .rstn(rstn),
    .in_valid(dp_in_valid), .in_ready(dp_in_ready),
    .in_warp(dp_warp), .in_op(dp_op), .in_imm(dp_imm),
    .out_valid(res_valid), .out_ready(res_ready),
    .out_warp(res_warp), .out_data(res_data)
  );
endmodule