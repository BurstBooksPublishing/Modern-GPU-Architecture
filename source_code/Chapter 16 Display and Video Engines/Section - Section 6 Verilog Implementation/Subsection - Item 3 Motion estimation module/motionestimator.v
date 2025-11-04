module motion_estimator #(
  parameter MB_SIZE = 16,
  parameter PIX_W = 8,
  parameter ADDR_W = 32,
  parameter SEARCH_W = 16
)(
  input  wire                   clk,
  input  wire                   rst_n,
  // control
  input  wire                   start,              // start search for next MB
  input  wire [ADDR_W-1:0]      ref_base_addr,      // top-left of search center (frame coords)
  // current block stream: one pixel per cycle, row-major, valid pulses
  input  wire [PIX_W-1:0]       cur_pixel_in,
  input  wire                   cur_pixel_valid,
  // reference RAM interface (synchronous read, 1-cycle latency)
  output reg  [ADDR_W-1:0]      ref_addr,
  output reg                    ref_rd,
  input  wire [PIX_W-1:0]       ref_pixel_in,
  // results
  output reg                    out_valid,
  output reg signed [15:0]      mv_x, mv_y,
  output reg  [31:0]            sad_min
);

localparam MB_PIXELS = MB_SIZE*MB_SIZE;
localparam SEARCH_DIM = 2*SEARCH_W+1;

reg [PIX_W-1:0] cur_block [0:MB_PIXELS-1];
integer load_idx;
reg loading;

// search indices
integer dy, dx;
integer px_idx;
reg [31:0] sad_acc;
reg [31:0] candidate_sad;
reg [15:0] best_dx, best_dy;
reg signed [15:0] signed_dx, signed_dy;

typedef enum reg [1:0] {IDLE=0, LOAD=1, SEARCH=2, DONE=3} state_t;
reg [1:0] state, next_state;

always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    state <= IDLE;
    loading <= 0;
    out_valid <= 0;
    ref_rd <= 0;
    ref_addr <= 0;
    sad_min <= 32'hFFFFFFFF;
    mv_x <= 0; mv_y <= 0;
  end else begin
    state <= next_state;
    case (state)
      IDLE: begin
        out_valid <= 0;
        sad_min <= 32'hFFFFFFFF;
        if (start) begin
          loading <= 1;
          load_idx <= 0;
        end
      end
      LOAD: begin
        if (cur_pixel_valid) begin
          cur_block[load_idx] <= cur_pixel_in; // store incoming pixels
          load_idx <= load_idx + 1;
          if (load_idx == MB_PIXELS-1) begin
            loading <= 0;
            // prepare search
            dy <= -SEARCH_W;
            dx <= -SEARCH_W;
            sad_min <= 32'hFFFFFFFF;
          end
        end
      end
      SEARCH: begin
        // issue read for current candidate pixel index
        if (px_idx < MB_PIXELS) begin
          // compute reference address: ref_base + candidate offset + pixel offset
          // here we assume linearized frame address; caller must map (x,y)->addr
          ref_addr <= ref_base_addr + ((dy + SEARCH_W)*MB_SIZE + (dx + SEARCH_W)); // placeholder mapping
          ref_rd <= 1;
          // read returns next cycle
        end else begin
          ref_rd <= 0;
        end
        // capture returned pixel and accumulate
        if (ref_rd) begin
          // absolute difference
          if (ref_pixel_in > cur_block[px_idx])
            sad_acc <= sad_acc + (ref_pixel_in - cur_block[px_idx]);
          else
            sad_acc <= sad_acc + (cur_block[px_idx] - ref_pixel_in);
          px_idx <= px_idx + 1;
          if (px_idx == MB_PIXELS-1) begin
            candidate_sad <= sad_acc;
            sad_acc <= 0;
            px_idx <= 0;
            // compare to best
            if (candidate_sad < sad_min) begin
              sad_min <= candidate_sad;
              best_dx <= dx;
              best_dy <= dy;
            end
            // advance candidate
            if (dx < SEARCH_W) dx <= dx + 1;
            else begin
              dx <= -SEARCH_W;
              if (dy < SEARCH_W) dy <= dy + 1;
              else begin
                // finished
                out_valid <= 1;
                mv_x <= best_dx;
                mv_y <= best_dy;
              end
            end
          end
        end
      end
      DONE: begin
        out_valid <= 0;
      end
    endcase
  end
end

// simple next-state logic
always @(*) begin
  next_state = state;
  case (state)
    IDLE: if (start) next_state = LOAD;
    LOAD: if (!loading) next_state = SEARCH;
    SEARCH: if (out_valid) next_state = DONE;
    DONE: next_state = IDLE;
  endcase
end

endmodule