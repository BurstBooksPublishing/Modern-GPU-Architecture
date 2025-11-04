module logical_rop_unit #(
  parameter WIDTH = 32  // total color width (default RGBA8 = 32)
)(
  input  wire                 clk,
  input  wire                 en,            // enable (registered externally)
  input  wire [WIDTH-1:0]     src,           // source color word
  input  wire [WIDTH-1:0]     dest,          // destination color word (from ROP cache)
  input  wire [3:0]           opcode,        // 4-bit opcode (16 functions)
  input  wire [3:0]           chan_mask,     // per-channel write mask, MSB=alpha
  output reg  [WIDTH-1:0]     result         // resulting color word
);
  // Byte mask expansion for channel lanes (assumes 8-bit channels)
  reg [WIDTH-1:0] mask_expanded;
  integer i;
  always @(*) begin
    mask_expanded = {WIDTH{1'b0}};
    for (i = 0; i < 4; i = i + 1) begin
      mask_expanded[i*8 +: 8] = {8{chan_mask[i]}}; // replicate mask to each byte
    end
  end

  // Compute raw logical result
  reg [WIDTH-1:0] raw;
  always @(*) begin
    case (opcode)
      4'h0: raw = {WIDTH{1'b0}};            // CLEAR (0)
      4'h1: raw = src & dest;               // AND
      4'h2: raw = src | dest;               // OR
      4'h3: raw = src ^ dest;               // XOR
      4'h4: raw = ~(src & dest);            // NAND
      4'h5: raw = ~(src | dest);            // NOR
      4'h6: raw = ~(src ^ dest);            // XNOR
      4'h7: raw = ~dest;                    // INVERT destination
      4'h8: raw = src;                      // COPY source
      4'h9: raw = dest;                     // NO-OP (keep dest)
      // remaining opcodes map to common combinations
      4'hA: raw = (src & ~dest);            // src AND NOT dest
      4'hB: raw = (~src & dest);            // NOT src AND dest
      4'hC: raw = src & ~ (src & dest);     // example custom func
      4'hD: raw = (src | ~dest);
      4'hE: raw = (~src) | dest;
      4'hF: raw = {WIDTH{1'b1}};            // SET (all ones)
      default: raw = dest;
    endcase
  end

  // Apply channel mask: only masked bytes take raw; others keep dest
  always @(posedge clk) begin
    if (en) result <= (raw & mask_expanded) | (dest & ~mask_expanded);
  end
endmodule