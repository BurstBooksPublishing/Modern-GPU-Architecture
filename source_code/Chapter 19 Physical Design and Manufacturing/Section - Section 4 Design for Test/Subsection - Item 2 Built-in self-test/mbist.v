module mbist_controller #(
  parameter ADDR_WIDTH = 10,
  parameter DATA_WIDTH = 32
)(
  input  wire                     clk,
  input  wire                     rst_n,
  input  wire                     start,      // start MBIST
  // single-port memory interface
  output reg  [ADDR_WIDTH-1:0]    mem_addr,
  output reg                      mem_en,
  output reg                      mem_we,
  output reg  [DATA_WIDTH-1:0]    mem_wdata,
  input  wire [DATA_WIDTH-1:0]    mem_rdata,
  output reg                      done,
  output reg                      fail
);

localparam STATE_IDLE      = 3'd0;
localparam STATE_W0_ALL    = 3'd1;
localparam STATE_R0_W1     = 3'd2;
localparam STATE_R1_VERIFY = 3'd3;
localparam STATE_DONE      = 3'd4;

reg [2:0] state;
reg [ADDR_WIDTH-1:0] addr;

// synchronous state machine
always @(posedge clk) begin
  if (!rst_n) begin
    state <= STATE_IDLE;
    addr  <= {ADDR_WIDTH{1'b0}};
    mem_en <= 1'b0;
    mem_we <= 1'b0;
    mem_wdata <= {DATA_WIDTH{1'b0}};
    mem_addr <= {ADDR_WIDTH{1'b0}};
    done <= 1'b0;
    fail <= 1'b0;
  end else begin
    case (state)
      STATE_IDLE: begin
        done <= 1'b0;
        fail <= 1'b0;
        if (start) begin
          addr <= {ADDR_WIDTH{1'b0}};
          state <= STATE_W0_ALL;
        end
      end

      // WRITE 0 to all addresses
      STATE_W0_ALL: begin
        mem_en <= 1'b1;
        mem_we <= 1'b1;
        mem_addr <= addr;
        mem_wdata <= {DATA_WIDTH{1'b0}};
        if (addr == {ADDR_WIDTH{1'b1}}) begin
          addr <= {ADDR_WIDTH{1'b0}};
          state <= STATE_R0_W1;
        end else addr <= addr + 1'b1;
      end

      // READ 0, then WRITE 1
      STATE_R0_W1: begin
        mem_en <= 1'b1;
        mem_we <= 1'b0;                // perform read
        mem_addr <= addr;
        // compare read data against 0 on next cycle (assumes 1-cycle read latency)
        if (mem_rdata != {DATA_WIDTH{1'b0}}) begin
          fail <= 1'b1;
          state <= STATE_DONE;
        end else begin
          // after successful read, write 1 to same address
          mem_en <= 1'b1;
          mem_we <= 1'b1;
          mem_wdata <= {DATA_WIDTH{1'b1}};
          mem_addr <= addr;
          if (addr == {ADDR_WIDTH{1'b1}}) begin
            addr <= {ADDR_WIDTH{1'b0}};
            state <= STATE_R1_VERIFY;
          end else addr <= addr + 1'b1;
        end
      end

      // READ 1 and verify
      STATE_R1_VERIFY: begin
        mem_en <= 1'b1;
        mem_we <= 1'b0;
        mem_addr <= addr;
        if (mem_rdata != {DATA_WIDTH{1'b1}}) begin
          fail <= 1'b1;
          state <= STATE_DONE;
        end else begin
          if (addr == {ADDR_WIDTH{1'b1}}) begin
            state <= STATE_DONE;
          end else addr <= addr + 1'b1;
        end
      end

      STATE_DONE: begin
        mem_en <= 1'b0;
        mem_we <= 1'b0;
        done <= 1'b1;
        state <= STATE_IDLE;
      end

      default: state <= STATE_IDLE;
    endcase
  end
end

endmodule