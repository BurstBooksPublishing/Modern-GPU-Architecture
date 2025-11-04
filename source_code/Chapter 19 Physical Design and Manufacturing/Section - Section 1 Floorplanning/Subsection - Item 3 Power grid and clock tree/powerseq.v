module power_island_seq(
  input  wire clk,            // always-on clock domain
  input  wire rst_n,          // active-low reset
  input  wire pwr_req,        // request to power-up island
  input  wire pwr_rail_ok,    // hardware power-rail good sensor
  output reg  pwr_en,         // enable power switch (PMOS/NMOS gate)
  output reg  iso_on,         // isolation cells enable
  output reg  ret_latch_en    // retention latch enable
);
  typedef enum reg [1:0] {OFF=2'b00, POWERING=2'b01, ON=2'b10} state_t;
  state_t state, next;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= OFF; else state <= next;
  end
  always @(*) begin
    // default outputs
    pwr_en = 1'b0; iso_on = 1'b1; ret_latch_en = 1'b0; next = state;
    case (state)
      OFF: if (pwr_req) next = POWERING;
      POWERING: begin
        pwr_en = 1'b1; // assert power switch
        if (pwr_rail_ok) begin
          iso_on = 1'b0; ret_latch_en = 1'b1; next = ON;
        end
      end
      ON: begin
        pwr_en = 1'b1; ret_latch_en = 1'b1; iso_on = 1'b0;
        if (!pwr_req) next = OFF;
      end
    endcase
  end
endmodule