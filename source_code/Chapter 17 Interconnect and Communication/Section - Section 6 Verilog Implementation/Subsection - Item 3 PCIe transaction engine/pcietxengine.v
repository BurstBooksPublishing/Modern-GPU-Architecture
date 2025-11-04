module pcie_tx_engine #(
  parameter TAG_WIDTH = 4,                 // number of tag bits
  parameter TAG_COUNT = (1< DATA_WIDTH/8) remain <= remain - DATA_WIDTH/8;
            else begin
              remain <= 16'd0;
              state <= S_IDLE;
              req_ready <= 1'b1;
            end
          end else tx_axis_tvalid <= 1'b0;
        end
      endcase
    end
  end

endmodule