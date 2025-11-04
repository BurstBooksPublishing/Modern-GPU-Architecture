module burst_adapter #(
  parameter DATA_IN_WIDTH = 256,      // internal bus width
  parameter MEM_WIDTH = 64,           // memory lane width
  parameter BURST_LEN = DATA_IN_WIDTH / MEM_WIDTH
)(
  input  wire                 clk,
  input  wire                 rst_n,
  // simple valid/ready handshake for input beat
  input  wire [DATA_IN_WIDTH-1:0] din,
  input  wire                 din_valid,
  output reg                  din_ready,
  // serialized memory lane output
  output reg [MEM_WIDTH-1:0]  mem_dout,
  output reg                  mem_valid,
  input  wire                 mem_ready
);

  // counters and register buffer
  reg [DATA_IN_WIDTH-1:0] buffer;
  reg [$clog2(BURST_LEN)-1:0] idx;
  reg                        active;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      din_ready <= 1'b1;
      mem_valid <= 1'b0;
      idx <= 0;
      active <= 1'b0;
      buffer <= {DATA_IN_WIDTH{1'b0}};
    end else begin
      // accept new beat when idle
      if (din_valid && din_ready) begin
        buffer <= din;
        active <= 1'b1;
        idx <= 0;
        din_ready <= 1'b0; // backpressure until burst done
      end
      // drive out serialized words
      if (active) begin
        mem_dout <= buffer[MEM_WIDTH-1:0];     // lowest bits first
        mem_valid <= 1'b1;
        if (mem_valid && mem_ready) begin
          // rotate buffer for next chunk
          buffer <= buffer >> MEM_WIDTH;
          idx <= idx + 1;
          if (idx == BURST_LEN-1) begin
            active <= 1'b0;
            mem_valid <= 1'b0;
            din_ready <= 1'b1;
          end
        end
      end else begin
        mem_valid <= 1'b0;
      end
    end
  end

endmodule