module mrt_write_arbiter #(
  parameter NUM_RT = 4,
  parameter DATA_WIDTH = 128,
  parameter ADDR_WIDTH = 32
)(
  input  wire                     clk,
  input  wire                     rstn,
  input  wire [NUM_RT-1:0]        valid_in,        // per-RT valid
  input  wire [NUM_RT*DATA_WIDTH-1:0] data_in,      // concat RT data
  input  wire [NUM_RT*ADDR_WIDTH-1:0]  addr_in,     // concat RT addr
  input  wire [NUM_RT-1:0]        mask_in,         // per-RT write mask
  output reg                      mem_valid,
  output reg [DATA_WIDTH-1:0]     mem_data,
  output reg [ADDR_WIDTH-1:0]     mem_addr,
  input  wire                     mem_ready
);
  reg [1:0] ptr; // round-robin pointer
  integer i;
  wire [NUM_RT-1:0] grant;
  // simple priority encoder starting at ptr
  reg [NUM_RT-1:0] rot_valid;
  always @(*) begin
    // rotate valid vector
    rot_valid = {valid_in, valid_in} >> ptr;
  end
  // grant calculation
  reg [NUM_RT-1:0] grant_unrot;
  always @(*) begin
    grant_unrot = 0;
    for (i=0;i> (NUM_RT - ptr));
  // selection and handshake
  always @(posedge clk) begin
    if (!rstn) begin
      mem_valid <= 0; mem_data <= 0; mem_addr <= 0; ptr <= 0;
    end else begin
      if (mem_valid && mem_ready) mem_valid <= 0; // cleared when accepted
      if (!mem_valid) begin
        for (i=0;i> (i*DATA_WIDTH);
            mem_addr  <= addr_in  >> (i*ADDR_WIDTH);
            ptr <= i + 1; // next start
          end
        end
      end
    end
  end
endmodule