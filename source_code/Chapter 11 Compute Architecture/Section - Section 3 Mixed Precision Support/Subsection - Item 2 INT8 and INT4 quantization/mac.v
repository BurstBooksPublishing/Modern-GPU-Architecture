module quant_serial_mac #(
  parameter integer N = 16,               // length of dot product (K)
  parameter integer A_W = 8,              // width of operand A (INT8 or INT4)
  parameter integer B_W = 8,              // width of operand B
  parameter integer ACC_W = 32            // accumulator width (>= A_W+B_W+ceil(log2(N)))
)(
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire                 start,                     // start pulse
  input  wire [N*A_W-1:0]     a_vec,                     // concatenated A lanes
  input  wire [N*B_W-1:0]     b_vec,                     // concatenated B lanes
  output reg  signed [ACC_W-1:0] acc_out,
  output reg                  done
);
  integer idx;
  reg [31:0] cnt;
  reg signed [ACC_W-1:0] acc_r;
  reg running;

  wire signed [A_W-1:0] a_lane = '0;
  wire signed [B_W-1:0] b_lane = '0;

  // extract current lane each cycle
  function automatic signed [A_W-1:0] get_a(input integer i);
    get_a = $signed(a_vec[i*A_W +: A_W]);
  endfunction
  function automatic signed [B_W-1:0] get_b(input integer i);
    get_b = $signed(b_vec[i*B_W +: B_W]);
  endfunction

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cnt <= 0;
      acc_r <= 0;
      acc_out <= 0;
      done <= 0;
      running <= 0;
    end else begin
      if (start && !running) begin
        running <= 1;
        cnt <= 0;
        acc_r <= 0;
        done <= 0;
      end else if (running) begin
        // multiply-accumulate one lane per cycle
        acc_r <= acc_r + (get_a(cnt) * get_b(cnt));
        cnt <= cnt + 1;
        if (cnt == N-1) begin
          running <= 0;
          acc_out <= acc_r + (get_a(N-1) * get_b(N-1));
          done <= 1;
        end
      end else begin
        done <= 0; // clear done until next run
      end
    end
  end
endmodule