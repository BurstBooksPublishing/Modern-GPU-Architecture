module trace_buffer #(parameter SAMPLE_DEPTH=1024, parameter WIDTH=64)
(
  input  wire                  clk,
  input  wire                  rst_n,
  input  wire                  arm,        // enable sampling
  input  wire                  trigger,    // latch context and stop
  input  wire [WIDTH-1:0]      sample_in,  // sampled signals (e.g., pc, warp id, mask)
  output reg                   done,       // capture finished
  input  wire [$clog2(SAMPLE_DEPTH)-1:0] read_addr, // readout address
  output wire [WIDTH-1:0]      read_data
);
  // memory storage (synthesizable)
  reg [WIDTH-1:0] mem [0:SAMPLE_DEPTH-1];
  reg [$clog2(SAMPLE_DEPTH)-1:0] write_ptr;
  reg capturing;

  // write-side logic: circular capture while armed and not done
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      write_ptr <= 0;
      done <= 1'b0;
      capturing <= 1'b0;
    end else begin
      if (arm && !capturing) capturing <= 1'b1;            // arm capture
      if (capturing && !done) begin
        mem[write_ptr] <= sample_in;                      // store sample
        write_ptr <= write_ptr + 1;
        if (trigger) begin
          done <= 1'b1;                                   // stop on trigger
          capturing <= 1'b0;
        end else if (write_ptr == SAMPLE_DEPTH-1) begin
          write_ptr <= 0;                                 // wrap (rolling)
        end
      end
      if (!arm) begin done <= 1'b0; end                     // reset done when disarmed
    end
  end

  assign read_data = mem[read_addr]; // simple synchronous/asynchronous read depends on tool
endmodule