module range_encoder #(
  parameter PROB_BITS = 11 // fixed-point probability resolution
) (
  input  wire        clk,
  input  wire        rst_n,
  // input symbol (1-bit) with valid/ready
  input  wire        in_valid,
  output reg         in_ready,
  input  wire        in_sym,         // symbol bit
  input  wire [PROB_BITS-1:0] in_p,  // probability of 1 in Q(PROB_BITS)
  // byte output (8-bit) with valid
  output reg  [7:0]  out_byte,
  output reg         out_valid,
  input  wire        out_ready
);
  // Internal state: low (32 bits), range (32 bits), buffer for pending bytes
  reg [31:0] low;
  reg [31:0] range;
  reg [7:0]  buffer;    // last emitted byte staging (not a FIFO)
  localparam R_MIN = 32'h01000000; // renorm threshold (~24-bit)
  // combinational probability multiply -> split
  wire [31:0] split; // range * p >> PROB_BITS
  assign split = (range * in_p) >> PROB_BITS;

  // Simple FSM: IDLE -> ENCODE -> RENORM -> OUTPUT
  typedef enum reg [1:0] {IDLE=0, ENCODE=1, RENORM=2} state_t;
  reg [1:0] state;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      low    <= 32'd0;
      range  <= 32'hFFFFFFFF;
      state  <= IDLE;
      in_ready <= 1'b1;
      out_valid <= 1'b0;
      out_byte <= 8'd0;
    end else begin
      case (state)
        IDLE: begin
          out_valid <= 1'b0;
          if (in_valid && in_ready) begin
            // perform binary range update
            if (in_sym) begin
              low <= low + (range - split);
              range <= split;
            end else begin
              range <= range - split;
            end
            in_ready <= 1'b0; // consume input for this cycle
            state <= RENORM;
          end
        end
        RENORM: begin
          // renormalize: emit top byte(s) while range < R_MIN
          if (range < R_MIN) begin
            // emit top byte from low >> 24
            if (!out_valid || (out_valid && out_ready)) begin
              out_byte <= low[31:24]; // output MSB
              out_valid <= 1'b1;
              // shift low and range
              low <= (low << 8);
              range <= (range << 8);
              // allow next input when one byte emitted
              in_ready <= 1'b1;
              state <= IDLE;
            end else begin
              // back-pressure: wait until consumer accepts byte
              out_valid <= out_valid;
            end
          end else begin
            // no renorm needed; ready to accept next input
            in_ready <= 1'b1;
            out_valid <= 1'b0;
            state <= IDLE;
          end
        end
      endcase
    end
  end
endmodule