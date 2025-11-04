module tess_iter_tri(
  input  wire        clk,
  input  wire        rstn,
  input  wire [7:0]  tess_level, // integer L
  input  wire        start,      // start generation
  output reg         valid,
  output reg  [7:0]  i_idx,      // barycentric i
  output reg  [7:0]  j_idx,      // barycentric j
  output reg         last
);
  // FSM states
  localparam IDLE=0, RUN=1;
  reg state;
  reg [7:0] L;
  reg [7:0] i, j;
  // start handling and clamping
  always @(posedge clk) begin
    if (!rstn) begin
      state<=IDLE; valid<=0; last<=0; i<=0; j<=0; L<=1;
    end else begin
      case(state)
      IDLE: begin
        valid<=0; last<=0;
        if (start) begin
          L <= (tess_level==0) ? 1 : tess_level; // clamp min 1
          i <= 0; j <= 0; state<=RUN;
        end
      end
      RUN: begin
        valid <= 1;
        i_idx <= i; j_idx <= j;
        // compute last when at final element
        last <= (i==L && j==0);
        // advance triangular counters: 0<=j<=L-i
        if (j < (L - i)) begin
          j <= j + 1;
        end else begin
          j <= 0;
          if (i < L) i <= i + 1;
          else state <= IDLE;
        end
      end
      endcase
    end
  end
endmodule