module ecc_secdec_64_8 (
  input  wire         clk,
  input  wire         rst,
  input  wire [63:0]  data_in,   // data returned from DRAM
  input  wire [7:0]   ecc_in,    // stored ECC bits from DRAM
  output reg  [63:0]  data_out,  // corrected data
  output reg          single_err,
  output reg          double_err
);
  // Masks must match controller/PHY H matrix (row-wise).
  parameter [63:0] MASK0 = 64'hFF00FF00FF00FF00;
  parameter [63:0] MASK1 = 64'h0F0F0F0F0F0F0F0F;
  parameter [63:0] MASK2 = 64'h3333333333333333;
  parameter [63:0] MASK3 = 64'h5555555555555555;
  parameter [63:0] MASK4 = 64'hAAAAAAAAAAAAAAAA;
  parameter [63:0] MASK5 = 64'hCCCCCCCCCCCCCCCC;
  parameter [63:0] MASK6 = 64'hF0F0F0F0F0F0F0F0;
  parameter [63:0] MASK7 = 64'hFFFFFFFFFFFFFFFF; // overall parity

  wire [7:0] exp_ecc;
  assign exp_ecc[0] = ^(data_in & MASK0);
  assign exp_ecc[1] = ^(data_in & MASK1);
  assign exp_ecc[2] = ^(data_in & MASK2);
  assign exp_ecc[3] = ^(data_in & MASK3);
  assign exp_ecc[4] = ^(data_in & MASK4);
  assign exp_ecc[5] = ^(data_in & MASK5);
  assign exp_ecc[6] = ^(data_in & MASK6);
  assign exp_ecc[7] = ^(data_in & MASK7);

  wire [7:0] syndrome = exp_ecc ^ ecc_in;

  // Locate single-bit error by mapping syndrome to bit index.
  // Simple binary-search LUT; real designs use optimized ROM.
  reg [6:0] err_bit_idx;
  reg       err_single_detected;
  integer   i;
  always @(*) begin
    err_single_detected = 1'b0;
    err_bit_idx = 7'd0;
    // scan data bits 0..63: compute syndrome if bit flipped
    for (i=0;i<64;i=i+1) begin
      // flipping bit i flips exp_ecc by XOR with ECC contribution mask
      // here compare precomputed mask effect by re-evaluating parity (synthesizable)
      if ((^( (data_in ^ (1'b1 << i)) & MASK0 ) ^ ^( (data_in ^ (1'b1 << i)) & MASK1 ) ^ 0) == 1'b0) begin end
      // For brevity, real implementation uses a syndrome-to-index ROM.
    end
  end

  // For this example, if nonzero syndrome and overall parity indicates single-bit:
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      data_out <= 64'b0;
      single_err <= 1'b0;
      double_err <= 1'b0;
    end else begin
      if (syndrome == 8'b0) begin
        data_out <= data_in;
        single_err <= 1'b0;
        double_err <= 1'b0;
      end else begin
        // Placeholder correction: in production use syndrome->index ROM then flip bit.
        // Mark detection flags conservatively.
        single_err <= 1'b0;
        double_err <= 1'b1;
        data_out <= data_in;
      end
    end
  end
endmodule