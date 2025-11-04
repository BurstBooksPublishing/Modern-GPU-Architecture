module opcode_dispatch (
  input  wire         clk,
  input  wire         rst_n,
  input  wire [31:0]  instr,      // instruction word
  input  wire         instr_valid,
  output reg          alu_valid,
  output reg  [31:0]  alu_instr,
  output reg          tmu_valid,
  output reg  [31:0]  tmu_instr,
  output reg          tensor_valid,
  output reg  [31:0]  tensor_instr
);
  // simple opcode map: bits [31:28] define unit
  wire [3:0] opcode = instr[31:28];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      alu_valid <= 0; tmu_valid <= 0; tensor_valid <= 0;
      alu_instr <= 32'b0; tmu_instr <= 32'b0; tensor_instr <= 32'b0;
    end else begin
      // default: clear outputs each cycle
      alu_valid <= 0; tmu_valid <= 0; tensor_valid <= 0;
      if (instr_valid) begin
        case (opcode)
          4'h0, 4'h1, 4'h2: begin // arithmetic opcodes
            alu_valid  <= 1;
            alu_instr  <= instr;
          end
          4'h8, 4'h9: begin // texture sampler ops
            tmu_valid  <= 1;
            tmu_instr  <= instr;
          end
          4'hC, 4'hD: begin // tensor ops
            tensor_valid <= 1;
            tensor_instr <= instr;
          end
          default: begin // treat as ALU by default
            alu_valid <= 1;
            alu_instr <= instr;
          end
        endcase
      end
    end
  end
endmodule