module plru4 (
  input  wire        clk,
  input  wire        rst,          // synchronous reset
  input  wire        access_valid, // valid access (hit or miss)
  input  wire [1:0]  access_way,   // way accessed when hit
  input  wire        miss_req,     // request victim (on miss)
  output reg  [1:0]  victim_way    // selected victim way
);
  reg [2:0] bits; // [2]=root, [1]=left, [0]=right

  // reset: prefer way0 as LRU
  always @(posedge clk) begin
    if (rst) begin
      bits <= 3'b000;
      victim_way <= 2'b00;
    end else begin
      if (access_valid) begin
        case (access_way)
          2'b00: begin bits[2]<=1'b1; bits[1]<=1'b1; end // accessed way0
          2'b01: begin bits[2]<=1'b1; bits[1]<=1'b0; end // accessed way1
          2'b10: begin bits[2]<=1'b0; bits[0]<=1'b1; end // accessed way2
          2'b11: begin bits[2]<=1'b0; bits[0]<=1'b0; end // accessed way3
        endcase
      end
      if (miss_req) begin
        // traverse tree to pick current LRU leaf
        if (bits[2]==1'b0) begin // left subtree LRU
          victim_way <= (bits[1]==1'b0) ? 2'b00 : 2'b01;
        end else begin // right subtree LRU
          victim_way <= (bits[0]==1'b0) ? 2'b10 : 2'b11;
        end
      end
    end
  end
endmodule