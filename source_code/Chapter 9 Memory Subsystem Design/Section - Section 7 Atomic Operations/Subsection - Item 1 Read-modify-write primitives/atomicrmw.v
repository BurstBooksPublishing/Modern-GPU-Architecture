module atomic_rmw_unit #(parameter ADDR_WIDTH=8, DATA_WIDTH=32, MEMWORDS=256)
(input clk, input rst,
 input req_valid, input [ADDR_WIDTH-1:0] req_addr,
 input [1:0] req_op, // 0=NOP,1=FETCH_ADD,2=CAS
 input [DATA_WIDTH-1:0] req_data, input [DATA_WIDTH-1:0] req_cmp,
 output reg req_ready,
 output reg resp_valid, output reg [DATA_WIDTH-1:0] resp_data);
  reg [DATA_WIDTH-1:0] mem [0:MEMWORDS-1];
  reg busy;
  reg [ADDR_WIDTH-1:0] locked_addr;
  // Simple single-cycle exclusive: lock, perform, unlock.
  always @(posedge clk) begin
    if (rst) begin
      req_ready <= 1'b1; resp_valid <= 1'b0; busy <= 1'b0;
    end else begin
      resp_valid <= 1'b0;
      if (req_valid && req_ready && !busy) begin
        busy <= 1'b1; locked_addr <= req_addr; req_ready <= 1'b0;
        if (req_op==1) begin // FETCH_ADD
          resp_data <= mem[req_addr];
          mem[req_addr] <= mem[req_addr] + req_data;
          resp_valid <= 1'b1;
          busy <= 1'b0; req_ready <= 1'b1;
        end else if (req_op==2) begin // CAS
          resp_data <= mem[req_addr];
          if (mem[req_addr]==req_cmp) mem[req_addr] <= req_data;
          resp_valid <= 1'b1;
          busy <= 1'b0; req_ready <= 1'b1;
        end else begin
          busy <= 1'b0; req_ready <= 1'b1;
        end
      end
    end
  end
endmodule