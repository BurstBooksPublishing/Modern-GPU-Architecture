module hetero_directory #(
  parameter N_CPU = 4,
  parameter N_GPU = 8,
  parameter LINE_ADDR_W = 32
)(
  input  wire                    clk,
  input  wire                    rst_n,
  // Request from CPU/GPU (simple arbiter assumed externally)
  input  wire [LINE_ADDR_W-1:0]  req_addr,
  input  wire [1:0]              req_type, // 0=Read,1=Write,2=ProbeAck
  input  wire                    req_valid,
  output reg                     req_ready,
  // Outputs: invalidation vector (one-hot per client group)
  output reg [N_CPU-1:0]         cpu_inval,
  output reg [N_GPU-1:0]         gpu_inval,
  output reg                     grant // directory granted ownership
);

  // Directory entry: state and sharer bitmasks
  typedef enum reg [1:0] {I=2'b00, S=2'b01, M=2'b10} state_t;
  reg [LINE_ADDR_W-1:0] tag;
  reg [N_CPU-1:0] cpu_sharers;
  reg [N_GPU-1:0] gpu_sharers;
  state_t state;

  // Simple single-line directory for example; real design uses CAM/RAM.
  always @(posedge clk) begin
    if (!rst_n) begin
      tag <= {LINE_ADDR_W{1'b0}};
      cpu_sharers <= {N_CPU{1'b0}};
      gpu_sharers <= {N_GPU{1'b0}};
      state <= I;
      cpu_inval <= {N_CPU{1'b0}};
      gpu_inval <= {N_GPU{1'b0}};
      grant <= 1'b0;
      req_ready <= 1'b1;
    end else begin
      cpu_inval <= {N_CPU{1'b0}};
      gpu_inval <= {N_GPU{1'b0}};
      grant <= 1'b0;
      if (req_valid && req_ready) begin
        if (req_type == 2'd0) begin // Read
          // move to shared, add requester handled externally
          state <= (state==M) ? M : S;
          grant <= 1'b1;
        end else if (req_type == 2'd1) begin // Write/upgrade
          // issue invalidations to all sharers
          cpu_inval <= cpu_sharers;
          gpu_inval <= gpu_sharers;
          // clear sharers, set owner later upon ack
          cpu_sharers <= {N_CPU{1'b0}};
          gpu_sharers <= {N_GPU{1'b0}};
          state <= M;
          grant <= 1'b1;
        end else if (req_type == 2'd2) begin // ProbeAck (sharer ack)
          // caller supplies which sharer cleared via sideband in real system
          // here we assume external updates; no-op for compact example
          grant <= 1'b0;
        end
      end
    end
  end
endmodule