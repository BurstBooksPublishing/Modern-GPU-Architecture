module ddn_router #(
  parameter IN_PORTS = 4,
  parameter OUT_PORTS = 2,
  parameter DATA_WIDTH = 32,
  parameter DEST_WIDTH = 1,
  parameter FIFO_ADDR = 3 // depth = 2^FIFO_ADDR
)(
  input  wire                       clk,
  input  wire                       rst,
  input  wire [(IN_PORTS*(DEST_WIDTH+DATA_WIDTH))-1:0] in_bus, // packed packets
  input  wire [IN_PORTS-1:0]        in_valid,
  output reg  [IN_PORTS-1:0]        in_ready,
  output reg  [OUT_PORTS-1:0]       out_valid,
  input  wire [OUT_PORTS-1:0]       out_ready,
  output reg  [(OUT_PORTS*DATA_WIDTH)-1:0] out_bus
);

localparam DEPTH = (1<0) begin
        out_bus[(j*DATA_WIDTH)+:DATA_WIDTH] <= fifo_mem[j][rd_ptr[j]];
        if (out_ready[j]) begin
          rd_ptr[j] <= rd_ptr[j] + 1;
          count[j] <= count[j] - 1;
          out_valid[j] <= 1;
        end else begin
          out_valid[j] <= 1;
        end
      end else begin
        out_valid[j] <= 0;
      end
    end
  end
end
endmodule