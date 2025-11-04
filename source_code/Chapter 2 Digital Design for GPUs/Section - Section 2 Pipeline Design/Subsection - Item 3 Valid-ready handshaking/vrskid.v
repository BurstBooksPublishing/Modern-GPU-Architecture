module vr_skid #(
  parameter WIDTH = 128
)(
  input  wire               clk,
  input  wire               rst_n,     // active-low sync reset
  // producer side
  input  wire               in_valid,
  input  wire [WIDTH-1:0]   in_data,
  output wire               in_ready,
  // consumer side
  output wire               out_valid,
  output wire [WIDTH-1:0]   out_data,
  input  wire               out_ready
);

  // stage register (drives output)
  reg                     stage_valid;
  reg [WIDTH-1:0]         stage_data;

  // skid register (holds one element when stage is blocked)
  reg                     skid_valid;
  reg [WIDTH-1:0]         skid_data;

  // combinational ready: accept if skid free and stage can advance
  assign in_ready  = (!skid_valid) && (!stage_valid || out_ready);
  assign out_valid = stage_valid;
  assign out_data  = stage_data;

  always @(posedge clk) begin
    if (!rst_n) begin
      stage_valid <= 1'b0;
      skid_valid  <= 1'b0;
    end else begin
      // normal advance: if stage empty, accept directly
      if (!stage_valid) begin
        if (in_valid) begin
          stage_valid <= 1'b1;
          stage_data  <= in_data;      // take input into stage
        end
      end else begin
        if (out_ready) begin
          // consumer consumed stage; refill from skid if present,
          // else take new input if offered this cycle
          if (skid_valid) begin
            stage_data  <= skid_data;
            stage_valid <= 1'b1;
            skid_valid  <= 1'b0;
          end else if (in_valid) begin
            stage_data  <= in_data;
            stage_valid <= 1'b1;
          end else begin
            stage_valid <= 1'b0;       // no data to hold
          end
        end else begin
          // stage is held; if producer provides and skid free, capture
          if (in_valid && !skid_valid) begin
            skid_data  <= in_data;
            skid_valid <= 1'b1;
          end
          // otherwise, remain held and possibly drop if skid full (rare)
        end
      end
    end
  end
endmodule