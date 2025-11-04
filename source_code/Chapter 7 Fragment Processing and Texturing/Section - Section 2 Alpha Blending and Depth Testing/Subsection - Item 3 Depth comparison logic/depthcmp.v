module depth_compare_unit #(
  parameter integer DEPTH_BITS = 24
)(
  input  wire                     clk,
  input  wire                     rstn,
  input  wire                     valid_in,           // fragment valid
  output reg                      ready_out,          // backpressure
  input  wire [DEPTH_BITS-1:0]    depth_frag,         // quantized frag depth
  input  wire [DEPTH_BITS-1:0]    depth_mem,          // current depth buffer value
  input  wire [2:0]               cmp_op,             // 3'b000=NEVER,...,3'b111=ALWAYS
  input  wire                     depth_write_enable, // allow depth updates
  output reg                      pass_out,           // test result
  output reg                      write_out,          // request depth write
  output reg  [DEPTH_BITS-1:0]    depth_out           // new depth value (registered)
);
  // combinational compare
  wire less  = (depth_frag <  depth_mem);
  wire equal = (depth_frag == depth_mem);
  wire greater = (depth_frag > depth_mem);

  reg pass_comb;
  always @(*) begin
    case (cmp_op)
      3'b000: pass_comb = 1'b0;               // NEVER
      3'b001: pass_comb = less;               // LESS
      3'b010: pass_comb = less || equal;      // LEQUAL
      3'b011: pass_comb = greater;            // GREATER
      3'b100: pass_comb = greater || equal;   // GEQUAL
      3'b101: pass_comb = equal;              // EQUAL
      3'b110: pass_comb = ~equal;             // NOTEQUAL
      3'b111: pass_comb = 1'b1;               // ALWAYS
      default: pass_comb = 1'b0;
    endcase
  end

  // Simple one-stage pipeline with valid-ready
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      ready_out <= 1'b1;
      pass_out  <= 1'b0;
      write_out <= 1'b0;
      depth_out <= {DEPTH_BITS{1'b0}};
    end else begin
      if (valid_in && ready_out) begin
        pass_out  <= pass_comb;
        // depth write only if test passes and writes enabled
        write_out <= pass_comb && depth_write_enable;
        depth_out <= depth_frag;
        ready_out <= 1'b0; // hold until consumer accepts (simple staging)
      end else begin
        // consumer must clear backpressure externally; for simplicity, become ready next cycle
        ready_out <= 1'b1;
      end
    end
  end
endmodule