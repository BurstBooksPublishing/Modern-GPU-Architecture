module bin2gray #(parameter WIDTH=8) (input  logic [WIDTH-1:0] bin, output logic [WIDTH-1:0] gray);
  assign gray = bin ^ (bin >> 1); // vector XOR and shift
endmodule

module gray2bin #(parameter WIDTH=8) (input logic [WIDTH-1:0] gray, output logic [WIDTH-1:0] bin);
  integer i;
  always_comb begin
    bin[WIDTH-1] = gray[WIDTH-1];
    for (i = WIDTH-2; i >= 0; i = i - 1)
      bin[i] = bin[i+1] ^ gray[i]; // iterative XOR per Eq. (2)
  end
endmodule

module sync_gray #(
  parameter WIDTH = 8
) (
  input  logic              clk_dst,
  input  logic              rst_n,
  input  logic [WIDTH-1:0]  gray_async, // from source domain
  output logic [WIDTH-1:0]  gray_sync   // synchronized to clk_dst
);
  // two-stage flip-flop synchronizer per bit (reduces metastability risk)
  logic [WIDTH-1:0] ff1, ff2;
  always_ff @(posedge clk_dst or negedge rst_n) begin
    if (!rst_n) begin
      ff1 <= '0;
      ff2 <= '0;
    end else begin
      ff1 <= gray_async;
      ff2 <= ff1;
    end
  end
  assign gray_sync = ff2;
endmodule

module async_write_pointer #(parameter WIDTH=8) (
  input  logic               clk_wr,
  input  logic               rst_n,
  input  logic               wr_en,
  output logic [WIDTH-1:0]   gray_wr   // sent asynchronously to read domain
);
  logic [WIDTH-1:0] bin_ptr;
  always_ff @(posedge clk_wr or negedge rst_n) begin
    if (!rst_n) bin_ptr <= '0;
    else if (wr_en) bin_ptr <= bin_ptr + 1'b1;
  end
  bin2gray #(.WIDTH(WIDTH)) b2g (.bin(bin_ptr), .gray(gray_wr));
endmodule