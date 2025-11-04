module context_fsm #(
  parameter WORDS = 256, parameter ADDR_WIDTH = 16
)(
  input  wire clk, rst,
  input  wire start_save, start_restore,
  output reg  busy, done,
  // simple memory-like interface
  output reg  mem_we, mem_re,
  output reg [ADDR_WIDTH-1:0] mem_addr,
  output reg [31:0] mem_wdata,
  input  wire [31:0] mem_rdata
);
  localparam IDLE=0, SAVE=1, RESTORE=2, WAIT=3;
  reg [1:0] state;
  reg [$clog2(WORDS)-1:0] idx;
  always @(posedge clk) begin
    if (rst) begin state<=IDLE; busy<=0; done<=0; mem_we<=0; mem_re<=0; idx<=0; end
    else begin
      done<=0;
      case(state)
        IDLE: begin
          busy <= 0;
          if (start_save) begin state<=SAVE; busy<=1; idx<=0; end
          else if (start_restore) begin state<=RESTORE; busy<=1; idx<=0; end
        end
        SAVE: begin // write sequential context words to memory
          mem_we <= 1; mem_addr <= idx; mem_wdata <= /* read regfile[idx] */ 32'hDEAD_BEEF;
          idx <= idx + 1;
          if (idx == WORDS-1) begin mem_we<=0; state<=WAIT; end
        end
        RESTORE: begin // read sequential context words from memory
          mem_re <= 1; mem_addr <= idx;
          state <= WAIT;
        end
        WAIT: begin
          mem_we <= 0; mem_re <= 0;
          // on a real bus, check ready/ack; here assume one-cycle
          if (start_restore) begin /* load mem_rdata -> regfile[idx] */ end
          idx <= idx + 1;
          if (idx == WORDS-1) begin state<=IDLE; busy<=0; done<=1; end
          else if (state==WAIT && start_save==0 && start_restore==0) state<=IDLE;
        end
      endcase
    end
  end
endmodule