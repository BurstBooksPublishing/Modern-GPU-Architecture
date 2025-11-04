module bypass_unit #(
  parameter REG_IDX_W = 5,          // register index width
  parameter DATA_W = 32
)(
  input  wire clk,
  input  wire [REG_IDX_W-1:0] src_idx,       // consumer source register
  input  wire [REG_IDX_W-1:0] ex_dest_idx,
  input  wire                  ex_valid,
  input  wire [DATA_W-1:0]     ex_value,
  input  wire [REG_IDX_W-1:0] mem_dest_idx,
  input  wire                  mem_valid,
  input  wire [DATA_W-1:0]     mem_value,
  input  wire [REG_IDX_W-1:0] wb_dest_idx,
  input  wire                  wb_valid,
  input  wire [DATA_W-1:0]     wb_value,
  input  wire [DATA_W-1:0]     rf_value,     // register file read (fallback)
  output reg  [DATA_W-1:0]     operand_out
);
  // Priority: EX -> MEM -> WB -> RF
  always @(*) begin
    if (ex_valid && (ex_dest_idx == src_idx))         operand_out = ex_value;
    else if (mem_valid && (mem_dest_idx == src_idx))  operand_out = mem_value;
    else if (wb_valid && (wb_dest_idx == src_idx))   operand_out = wb_value;
    else                                               operand_out = rf_value;
  end
endmodule