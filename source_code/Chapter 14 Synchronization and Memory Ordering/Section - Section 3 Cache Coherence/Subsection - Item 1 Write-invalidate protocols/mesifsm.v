module mesi_line(
  input  wire        clk, rst,
  input  wire        cpu_read, cpu_write,      // core requests
  input  wire        remote_inval,             // incoming invalidation
  output reg         issue_inval_bcast,        // request directory to invalidate others
  output reg         grant_exclusive,          // grant to local core
  output reg         writeback_req             // request writeback to L2
);
  // State encoding
  localparam I=2'b00, S=2'b01, E=2'b10, M=2'b11;
  reg [1:0] state, next_state;

  // State transition combinational logic
  always @(*) begin
    issue_inval_bcast = 1'b0;
    grant_exclusive   = 1'b0;
    writeback_req     = 1'b0;
    next_state = state;
    case (state)
      I: begin
        if (cpu_read) next_state = S;                // cold read -> shared after fill
        else if (cpu_write) begin
          issue_inval_bcast = 1'b1;                  // ask directory for exclusive
          // stay in I until grant; grant_exclusive asserted by directory (external)
        end
      end
      S: begin
        if (cpu_write) begin
          issue_inval_bcast = 1'b1;                  // upgrade to exclusive
          // wait for grant
        end else if (remote_inval) next_state = I;   // another core upgraded -> invalidate
      end
      E: begin
        if (cpu_write) next_state = M;               // first local write -> modified
        else if (remote_inval) begin next_state = I; writeback_req = 1'b1; end
      end
      M: begin
        if (remote_inval) begin
          writeback_req = 1'b1;                      // supply dirty data to directory/L2
          next_state = I;
        end
      end
      default: next_state = I;
    endcase
  end

  // Sequential update (grant_exclusive asserted externally when directory grants)
  always @(posedge clk or posedge rst) begin
    if (rst) state <= I;
    else begin
      // simulate grant handshake: directory sets grant_exclusive externally.
      if (grant_exclusive) state <= E;
      else state <= next_state;
    end
  end
endmodule