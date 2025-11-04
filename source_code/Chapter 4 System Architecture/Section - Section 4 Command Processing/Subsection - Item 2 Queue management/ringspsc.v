module ring_queue #(
  parameter WIDTH = 64,              // command width
  parameter LOG_DEPTH = 6            // depth = 2^LOG_DEPTH
)(
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire                 push, // producer valid
  input  wire [WIDTH-1:0]     din,
  output wire                 full,
  input  wire                 pop,  // consumer ready
  output wire [WIDTH-1:0]     dout,
  output wire                 empty,
  output wire [LOG_DEPTH+1:0] count // used slots
);
  localparam DEPTH = 1 << LOG_DEPTH;
  reg [WIDTH-1:0] mem [0:DEPTH-1];
  reg [LOG_DEPTH:0] head, tail; // extra bit for full detection

  // write path
  wire [$clog2(DEPTH)-1:0] tail_idx = tail[LOG_DEPTH-1:0];
  wire [$clog2(DEPTH)-1:0] head_idx = head[LOG_DEPTH-1:0];

  assign full  = ( (tail[LOG_DEPTH] != head[LOG_DEPTH]) &&
                   (tail_idx == head_idx) );
  assign empty = (tail == head);
  assign dout  = mem[head_idx];
  assign count = (tail - head); // arithmetic modulo with extra bit

  always @(posedge clk) begin
    if (!rst_n) begin
      head <= 0;
      tail <= 0;
    end else begin
      if (push && !full) begin
        mem[tail_idx] <= din;        // enqueue
        tail <= tail + 1;
      end
      if (pop && !empty) begin
        head <= head + 1;            // dequeue
      end
    end
  end
endmodule