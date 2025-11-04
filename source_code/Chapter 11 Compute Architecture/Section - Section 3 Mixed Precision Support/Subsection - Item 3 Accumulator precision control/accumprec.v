module accumulator_precision_ctrl
  #(
    parameter DATA_IN = 16,          // input product width
    parameter ACCW    = 48,          // accumulator width
    parameter OUT_W   = 32,          // output width after rounding
    parameter LFSR_W  = 16
  )
  (
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   in_valid,
    input  wire signed [DATA_IN-1:0] addend_in, // incoming signed product
    input  wire [1:0]             rnd_mode,     // 00=truncate,01=RN,10=stochastic
    input  wire [5:0]             shift_amt,    // right-shift amount on commit
    input  wire                   commit,       // produce out_data
    output reg                    out_valid,
    output reg signed [OUT_W-1:0] out_data
  );

  // accumulator register
  reg signed [ACCW-1:0] acc_reg;
  wire signed [ACCW-1:0] addext = {{(ACCW-DATA_IN){addend_in[DATA_IN-1]}}, addend_in};

  // simple LFSR for stochastic rounding
  reg [LFSR_W-1:0] lfsr;
  wire [LFSR_W-1:0] lfsr_next = {lfsr[LFSR_W-2:0], lfsr[LFSR_W-1] ^ lfsr[2]};

  // rounding bias computation
  reg signed [ACCW-1:0] bias;
  integer sa;
  always @(*) begin
    sa = shift_amt;
    if (sa == 0) begin
      bias = 0;
    end else begin
      case (rnd_mode)
        2'b01: begin // round-to-nearest: add +2^{sa-1} for positive, -2^{sa-1} for negative
          bias = (acc_reg >= 0) ? ({{(ACCW-sa-1){1'b0}}, 1'b1, {(sa-1){1'b0}}})
                                : -({{(ACCW-sa-1){1'b0}}, 1'b1, {(sa-1){1'b0}}});
        end
        2'b10: begin // stochastic: add random integer in [0,2^{sa}-1]
          bias = {{(ACCW-sa){1'b0}}, lfsr[sa-1:0]}; // treat as positive fraction
        end
        default: bias = 0; // truncate
      endcase
    end
  end

  // accumulation and LFSR update
  always @(posedge clk) begin
    if (rst) begin
      acc_reg <= 0;
      lfsr   <= {LFSR_W{1'b1}};
      out_valid <= 0;
      out_data  <= 0;
    end else begin
      if (in_valid) acc_reg <= acc_reg + addext;
      lfsr <= lfsr_next;
      if (commit) begin
        // biased rounding then arithmetic shift
        if (shift_amt == 0) begin
          // direct saturation to OUT_W
          if (acc_reg > $signed({{(ACCW-OUT_W){1'b0}}, {1'b0, {(OUT_W-1){1'b1}}}}))
            out_data <= $signed({1'b0, {(OUT_W-1){1'b1}}});
          else if (acc_reg < -($signed({1'b0, {(OUT_W-1){1'b1}}}) + 1))
            out_data <= -($signed({1'b0, {(OUT_W-1){1'b1}}}) + 1);
          else
            out_data <= acc_reg[OUT_W-1:0];
        end else begin
          // add bias then arithmetic shift
          reg signed [ACCW-1:0] tmp;
          tmp = acc_reg + bias;
          out_data <= tmp >>> shift_amt; // arithmetic right shift
        end
        out_valid <= 1'b1;
      end else begin
        out_valid <= 1'b0;
      end
    end
  end

endmodule