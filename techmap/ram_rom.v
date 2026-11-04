// Memories with one asynchronous read and no writes become a tiled read-only array (placement_engine/ram.py), its
// contents from the memory's initial value. Rows past the memory's size read 0.
(* techmap_celltype = "$mem_v2" *)
module _ram_rom (RD_CLK, RD_EN, RD_ARST, RD_SRST, RD_ADDR, RD_DATA, WR_CLK, WR_EN, WR_ADDR, WR_DATA);
    parameter MEMID = "";
    parameter SIZE = 0;
    parameter OFFSET = 0;
    parameter ABITS = 0;
    parameter WIDTH = 0;
    parameter INIT = 0;
    parameter RD_PORTS = 0;
    parameter RD_WIDE_CONTINUATION = 0;
    parameter RD_CLK_ENABLE = 0;
    parameter RD_CLK_POLARITY = 0;
    parameter RD_TRANSPARENCY_MASK = 0;
    parameter RD_COLLISION_X_MASK = 0;
    parameter RD_CE_OVER_SRST = 0;
    parameter RD_INIT_VALUE = 0;
    parameter RD_ARST_VALUE = 0;
    parameter RD_SRST_VALUE = 0;
    parameter WR_PORTS = 0;
    parameter WR_WIDE_CONTINUATION = 0;
    parameter WR_CLK_ENABLE = 0;
    parameter WR_CLK_POLARITY = 0;
    parameter WR_PRIORITY_MASK = 0;

    input [RD_PORTS-1:0] RD_CLK, RD_EN, RD_ARST, RD_SRST;
    input [RD_PORTS*ABITS-1:0] RD_ADDR;
    output [RD_PORTS*WIDTH-1:0] RD_DATA;
    input [WR_PORTS-1:0] WR_CLK;
    input [WR_PORTS*WIDTH-1:0] WR_EN, WR_DATA;
    input [WR_PORTS*ABITS-1:0] WR_ADDR;

    wire _TECHMAP_FAIL_ = RD_PORTS != 1 || WR_PORTS != 0 || RD_CLK_ENABLE !== 1'b0 || OFFSET != 0 || ABITS < 1
        || SIZE > 1 << ABITS || RD_WIDE_CONTINUATION !== 1'b0;

    localparam DEPTH = 1 << ABITS;
    generate
        if (RD_PORTS == 1) begin : map
            ram_rom #(.DEPTH(DEPTH), .ABITS(ABITS), .WIDTH(WIDTH), .INIT(INIT)) _TECHMAP_REPLACE_ (
                .aa(RD_ADDR[ABITS-1:0]), .ra(RD_DATA[WIDTH-1:0]));
        end
    endgenerate
endmodule
