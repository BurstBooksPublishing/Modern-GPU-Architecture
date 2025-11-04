module tessellator #(
  parameter MAX_LEVEL = 64
)(
  input  wire         clk,
  input  wire         rst_n,
  // control: factors in Q16.16 fixed point
  input  wire [31:0]  outer0, outer1, outer2, inner0, inner1,
  input  wire [1:0]   prim_mode, // 0=triangle,1=quad
  input  wire         cfg_valid,
  output reg          cfg_ready,
  // vertex stream handshake
  output reg  [31:0]  coord0, coord1, coord2, // Q16.16: barycentric (tri) or (u,v,unused) (quad)
  output reg          out_valid,
  input  wire         out_ready
);

  // internal regs
  reg [7:0] level; // integer tessellation level N
  reg [15:0] recipN; // reciprocal in Q0.16 for simple multiply (unused for clarity)
  reg [7:0] i,j; // loop counters
  reg [1:0] state;
  localparam IDLE=0, LOAD=1, GEN=2, DONE=3;

  // config stage: round outer0 to integer level (simple policy: round nearest)
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state<=IDLE; cfg_ready<=1'b0; out_valid<=1'b0;
    end else begin
      case(state)
        IDLE: begin
          out_valid<=1'b0; cfg_ready<=1'b1;
          if (cfg_valid) begin
            cfg_ready<=1'b0; state<=LOAD;
            // simple round to nearest integer: (x + 0.5) trunc
            // outer0 is Q16.16
            level <= (outer0 + 32'h00008000) >> 16;
            if (level > MAX_LEVEL) level <= MAX_LEVEL;
            if (level == 0) level <= 1;
          end
        end
        LOAD: begin
          // prepare counters
          i<=0; j<=0;
          // compute reciprocal N in Q0.16 for later multiply (optional)
          recipN <= (65536 + level/2) / level; // approximate reciprocal
          state<=GEN;
        end
        GEN: begin
          if (!out_valid || (out_valid && out_ready)) begin
            // generate barycentric for triangle patch
            if (prim_mode==0) begin
              // when i+j > N, advance i and j appropriately
              if (i + j > level) begin
                j <= 0;
                i <= i + 1;
              end else begin
                // compute barycentric b0=(N-i-j)/N, b1=i/N, b2=j/N using integer mul
                coord0 <= ((level - i - j) * 65536) / level; // Q16.16
                coord1 <= (i * 65536) / level;
                coord2 <= (j * 65536) / level;
                out_valid <= 1'b1;
                // advance j for next
                j <= j + 1;
                // check termination
                if (i > level) state<=DONE;
              end
            end else begin
              // quad topology simplified: emit regular grid (u,v)
              coord0 <= ((i * 65536) / level); // u
              coord1 <= ((j * 65536) / level); // v
              coord2 <= 32'd0;
              out_valid <= 1'b1;
              // advance grid counters
              if (j == level) begin j<=0; i<=i+1; end else j<=j+1;
              if (i > level) state<=DONE;
            end
          end
        end
        DONE: begin
          out_valid<=1'b0; state<=IDLE;
        end
      endcase
    end
  end

endmodule