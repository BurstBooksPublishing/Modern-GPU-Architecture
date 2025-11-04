module pim_mac #(
  parameter ADDR_W = 16, DATA_W = 32, LEN_W = 16
)(
  input  wire                   clk,
  input  wire                   rst,
  input  wire                   start,         // start command
  input  wire [ADDR_W-1:0]      base_addr,     // base of vector pair
  input  wire [LEN_W-1:0]       length,        // number of elements
  // simple memory port A (vector A)
  output reg  [ADDR_W-1:0]      rd_addr_a,
  output reg                    rd_en_a,
  input  wire [DATA_W-1:0]      rd_data_a,
  // simple memory port B (vector B)
  output reg  [ADDR_W-1:0]      rd_addr_b,
  output reg                    rd_en_b,
  input  wire [DATA_W-1:0]      rd_data_b,
  // result writeback port
  output reg  [ADDR_W-1:0]      wr_addr,
  output reg                    wr_en,
  output reg  [DATA_W-1:0]      wr_data,
  output reg                    busy
);
  reg [LEN_W-1:0] cnt;
  reg signed [63:0] acc; // accumulator wider than DATA_W for safe MAC
  localparam IDLE=0, READ=1, ACC=2, WRITE=3;
  reg [1:0] state;
  always @(posedge clk) begin
    if (rst) begin
      state <= IDLE; busy <= 0; rd_en_a<=0; rd_en_b<=0; wr_en<=0;
    end else begin
      case(state)
        IDLE: begin
          if (start) begin
            cnt <= length;
            rd_addr_a <= base_addr; rd_addr_b <= base_addr + length; // layout: A then B
            acc <= 0; busy <= 1;
            state <= READ;
          end
        end
        READ: begin
          if (cnt != 0) begin
            rd_en_a <= 1; rd_en_b <= 1;
            state <= ACC;
          end else begin
            rd_en_a <= 0; rd_en_b <= 0; state <= WRITE;
          end
        end
        ACC: begin
          rd_en_a <= 0; rd_en_b <= 0;
          acc <= acc + $signed(rd_data_a) * $signed(rd_data_b);
          rd_addr_a <= rd_addr_a + 1; rd_addr_b <= rd_addr_b + 1;
          cnt <= cnt - 1;
          state <= READ;
        end
        WRITE: begin
          wr_addr <= base_addr - 1; wr_data <= acc[DATA_W-1:0]; wr_en <= 1;
          state <= IDLE; busy <= 0;
        end
      endcase
    end
  end
endmodule