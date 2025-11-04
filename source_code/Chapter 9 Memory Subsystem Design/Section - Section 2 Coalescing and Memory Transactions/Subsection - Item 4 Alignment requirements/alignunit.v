module alignment_unit #(
  parameter ADDR_WIDTH = 64,
  parameter LINE_BYTES = 128  // must be power-of-two
) (
  input  wire [ADDR_WIDTH-1:0] addr,     // byte address
  input  wire [31:0]           size,     // requested size in bytes
  output wire [ADDR_WIDTH-1:0] aligned_base, // addr aligned down to LINE_BYTES
  output wire [31:0]           units,    // number of LINE_BYTES units touched
  output wire                   crosses  // high if spans >1 unit
);
  localparam LOG_LINE = $clog2(LINE_BYTES);
  wire [LOG_LINE-1:0] offset = addr[LOG_LINE-1:0]; // within-line offset
  assign aligned_base = {addr[ADDR_WIDTH-1:LOG_LINE], {LOG_LINE{1'b0}}};
  // units = ceil((offset + size)/LINE_BYTES)
  wire [LOG_LINE+31:0] sum = { {(LOG_LINE){1'b0}}, size } + { {(31){1'b0}}, offset };
  assign units = (sum + (LINE_BYTES-1)) >> LOG_LINE;
  assign crosses = (units != 32'd1);
endmodule