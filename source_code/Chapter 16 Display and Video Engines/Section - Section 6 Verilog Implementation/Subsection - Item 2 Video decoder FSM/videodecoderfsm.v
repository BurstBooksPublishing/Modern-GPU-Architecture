module video_decoder_fsm #(
  parameter IDLE = 3'd0,
  parameter FETCH = 3'd1,
  parameter PARSE = 3'd2,
  parameter ENTROPY = 3'd3,
  parameter MC = 3'd4,
  parameter WRITEOUT = 3'd5,
  parameter ERROR = 3'd6
)(
  input  wire        clk,
  input  wire        rst_n,
  input  wire        start,          // request to decode a frame
  input  wire        dma_ack,        // DMA accepted
  input  wire        fetch_done,     // fetch completed
  input  wire        parse_done,     // parser finished
  input  wire        entropy_done,   // entropy decoder finished
  input  wire        mc_done,        // motion compensation finished
  input  wire        write_done,     // output write finished
  input  wire        stall,          // backpressure from memory/display
  input  wire        flush,          // abort & flush
  output reg         dma_req,        // request DMA fetch
  output reg         parse_start,    // start parser
  output reg         entropy_start,  // start entropy decoder
  output reg         mc_start,       // start motion compensation
  output reg         write_start,    // start frame write
  output reg         done,           // frame done
  output reg  [2:0]  state_out       // debug state
);

  reg [2:0] state, next_state;
  reg [3:0] dma_credits; // prevent issuing too many DMA bursts

  // next-state combinational logic
  always @(*) begin
    // default deasserts
    dma_req = 1'b0;
    parse_start = 1'b0;
    entropy_start = 1'b0;
    mc_start = 1'b0;
    write_start = 1'b0;
    done = 1'b0;
    next_state = state;
    case (state)
      IDLE: if (start && !stall) begin
              dma_req = 1'b1; // fetch header/bitstream
              next_state = FETCH;
            end
      FETCH: if (fetch_done) begin
               parse_start = 1'b1;
               next_state = PARSE;
             end else if (flush) next_state = IDLE;
      PARSE: if (parse_done) begin
               entropy_start = 1'b1;
               next_state = ENTROPY;
             end else if (flush) next_state = IDLE;
      ENTROPY: if (entropy_done) begin
                 mc_start = 1'b1;
                 next_state = MC;
               end else if (flush) next_state = IDLE;
      MC: if (mc_done) begin
            write_start = 1'b1;
            next_state = WRITEOUT;
          end else if (flush) next_state = IDLE;
      WRITEOUT: if (write_done) begin
                 done = 1'b1;
                 next_state = IDLE;
               end else if (flush) next_state = IDLE;
      ERROR: begin
               // wait for reset or explicit restart
               if (flush) next_state = IDLE;
             end
      default: next_state = IDLE;
    endcase
  end

  // sequential updates
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      dma_credits <= 4'd0;
    end else begin
      state <= next_state;
      // dma credit management when dma_ack asserted
      if (dma_req && dma_ack) dma_credits <= dma_credits + 1'b1;
      if (fetch_done && dma_credits != 0) dma_credits <= dma_credits - 1'b1;
    end
  end

  // expose debug state
  always @(posedge clk) state_out <= state;

endmodule