module mem_sched #(
  parameter ADDR_W = 32,
  parameter QD = 8,               // queue depth
  parameter BANKS = 8
) (
  input  wire clk, rst,
  input  wire req_valid,          // single enqueue interface (simplified)
  input  wire req_rw,             // 1=write,0=read
  input  wire [ADDR_W-1:0] req_addr,
  output reg  grant_valid,
  output reg  grant_rw,
  output reg  [ADDR_W-1:0] grant_addr
);
  // simple FIFO storage
  reg [ADDR_W-1:0] addr_q [0:QD-1];
  reg              rw_q   [0:QD-1];
  reg  [3:0] head, tail, count;

  // bank open-row table (simple), stores row index or -1 when closed
  reg [15:0] open_row [0:BANKS-1]; // assume row fits 16 bits
  integer i;

  // enqueue
  always @(posedge clk) begin
    if (rst) begin head<=0; tail<=0; count<=0; end
    else if (req_valid && count < QD) begin
      addr_q[tail] <= req_addr;
      rw_q[tail]   <= req_rw;
      tail <= tail + 1;
      count <= count + 1;
    end
  end

  // simple comb comparator to find a row-hit in queue
  reg hit_found; reg [3:0] hit_idx; reg [15:0] req_row; reg [2:0] req_bank;
  always @(*) begin
    hit_found = 0; hit_idx = 0;
    for (i=0;i0) begin
      if (hit_found) begin
        grant_valid <= 1;
        grant_addr  <= addr_q[hit_idx];
        grant_rw    <= rw_q[hit_idx];
        // remove entry by shifting (simple, for demo)
        for (i=hit_idx;i!=tail;i=(i+1)%QD) begin
          addr_q[i] <= addr_q[(i+1)%QD];
          rw_q[i]   <= rw_q[(i+1)%QD];
        end
        tail <= tail - 1; count <= count - 1;
      end else begin
        grant_valid <= 1;
        grant_addr  <= addr_q[head];
        grant_rw    <= rw_q[head];
        head <= head + 1; count <= count - 1;
      end
      // update open_row for write/activate canonically (not modeled)
    end else grant_valid <= 0;
  end
endmodule