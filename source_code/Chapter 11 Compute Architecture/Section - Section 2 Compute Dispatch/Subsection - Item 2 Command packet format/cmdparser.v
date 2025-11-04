module cmd_packet_parser (
  input  wire        clk,
  input  wire        rst_n,
  // input stream (AXI4-Stream style)
  input  wire        in_valid,
  output reg         in_ready,
  input  wire [31:0] in_data,
  input  wire        in_last, // marks end of payload (optional)
  // decoded metadata output
  output reg         meta_valid,
  input  wire        meta_ready,
  output reg [15:0]  kernel_id,
  output reg [31:0]  grid_x, grid_y, grid_z,
  output reg [15:0]  block_x, block_y, block_z,
  output reg [31:0]  shared_bytes,
  output reg [47:0]  arg_ptr
);
  // FSM states
  localparam IDLE=0, HDR=1, PAYLOAD=2, DONE=3;
  reg [1:0] state;
  reg [3:0] hdr_cnt;
  reg [31:0] tmp_word0;
  reg [31:0] tmp_word1;

  // Header word count fixed to 7 for this parser
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE; in_ready <= 1'b0; hdr_cnt <= 0;
      meta_valid <= 1'b0;
    end else begin
      case (state)
        IDLE: begin
          meta_valid <= 1'b0;
          if (in_valid) begin
            in_ready <= 1'b1; hdr_cnt <= 1;
            tmp_word0 <= in_data; // opcode/flags
            state <= HDR;
          end else begin
            in_ready <= 1'b0;
          end
        end
        HDR: begin
          if (in_valid && in_ready) begin
            hdr_cnt <= hdr_cnt + 1;
            case (hdr_cnt)
              1: begin tmp_word1 <= in_data; kernel_id <= in_data[15:0]; end
              2: grid_x <= in_data;
              3: grid_y <= in_data;
              4: grid_z <= in_data;
              5: begin block_x <= in_data[11:0]; block_y <= in_data[23:12]; block_z <= in_data[31:24]; end
              6: shared_bytes <= in_data;
              7: begin arg_ptr[31:0] <= in_data; end
            endcase
            if (hdr_cnt==7) begin
              state <= PAYLOAD; in_ready <= 1'b1;
            end
          end
        end
        PAYLOAD: begin
          // consume or forward payload; here we just wait for in_last
          if (in_valid && in_last) begin
            in_ready <= 1'b0;
            state <= DONE;
          end
        end
        DONE: begin
          if (!meta_valid && meta_ready) begin
            // second half of arg_ptr would be provided by another header word in real packet
            meta_valid <= 1'b1;
            state <= IDLE;
          end
        end
      endcase
    end
  end
endmodule