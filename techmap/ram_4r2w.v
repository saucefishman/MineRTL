// Memories with four asynchronous reads and two rising edge writes become banks of tiled arrays (placement_engine/ram.py).
// Each bank has one write path: the ports take turns in the clock's low phase, and the second port wins a clash.
// TODO: other port counts, partial writes.
(* techmap_celltype = "$mem_v2" *)
module _ram_4r2w (RD_CLK, RD_EN, RD_ARST, RD_SRST, RD_ADDR, RD_DATA, WR_CLK, WR_EN, WR_ADDR, WR_DATA);
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
    parameter _TECHMAP_BITS_CONNMAP_ = 0;
    parameter _TECHMAP_CONNMAP_WR_EN_ = 0;

    input [RD_PORTS-1:0] RD_CLK, RD_EN, RD_ARST, RD_SRST;
    input [RD_PORTS*ABITS-1:0] RD_ADDR;
    output [RD_PORTS*WIDTH-1:0] RD_DATA;
    input [WR_PORTS-1:0] WR_CLK;
    input [WR_PORTS*WIDTH-1:0] WR_EN, WR_DATA;
    input [WR_PORTS*ABITS-1:0] WR_ADDR;

    localparam CONN = _TECHMAP_BITS_CONNMAP_;
    function automatic whole_word_writes(input integer unused);
        integer p, i;
        begin
            whole_word_writes = 1;
            for (p = 0; p < WR_PORTS; p = p + 1)
                for (i = 1; i < WIDTH; i = i + 1)
                    if (_TECHMAP_CONNMAP_WR_EN_[(p*WIDTH+i)*CONN +: CONN] !== _TECHMAP_CONNMAP_WR_EN_[p*WIDTH*CONN +: CONN])
                        whole_word_writes = 0;
        end
    endfunction

    wire _TECHMAP_FAIL_ = RD_PORTS != 4 || WR_PORTS != 2 || RD_CLK_ENABLE !== 4'b0000 || WR_CLK_ENABLE !== 2'b11
        || WR_CLK_POLARITY !== 2'b11 || OFFSET != 0 || ABITS < 1 || SIZE != 1 << ABITS || RD_WIDE_CONTINUATION !== 4'b0000
        || WR_WIDE_CONTINUATION !== 2'b00 || (WR_PRIORITY_MASK !== 4'b0100 && WR_PRIORITY_MASK !== 4'b0000) || !whole_word_writes(0);

    localparam ROW_BITS = ABITS < 6 ? ABITS : 6;
    localparam BANK_BITS = ABITS - ROW_BITS;
    localparam ROWS = 1 << ROW_BITS;
    localparam BANKS = 1 << BANK_BITS;

    wire clk = WR_CLK[0];
    wire [ABITS-1:0] waddr0, waddr1;
    wire [WIDTH-1:0] wdata0, wdata1;
    wire wen0, wen1;
    \$dff #(.WIDTH(ABITS), .CLK_POLARITY(1'b1)) waddr0_reg (.CLK(clk), .D(WR_ADDR[ABITS-1:0]), .Q(waddr0));
    \$dff #(.WIDTH(WIDTH), .CLK_POLARITY(1'b1)) wdata0_reg (.CLK(clk), .D(WR_DATA[WIDTH-1:0]), .Q(wdata0));
    \$dff #(.WIDTH(1), .CLK_POLARITY(1'b1)) wen0_reg (.CLK(clk), .D(WR_EN[0]), .Q(wen0));
    \$dff #(.WIDTH(ABITS), .CLK_POLARITY(1'b1)) waddr1_reg (.CLK(clk), .D(WR_ADDR[WR_PORTS*ABITS-1:(WR_PORTS-1)*ABITS]), .Q(waddr1));
    \$dff #(.WIDTH(WIDTH), .CLK_POLARITY(1'b1)) wdata1_reg (.CLK(clk), .D(WR_DATA[WR_PORTS*WIDTH-1:(WR_PORTS-1)*WIDTH]), .Q(wdata1));
    \$dff #(.WIDTH(1), .CLK_POLARITY(1'b1)) wen1_reg (.CLK(clk), .D(WR_EN[(WR_PORTS-1)*WIDTH]), .Q(wen1));

    // After clk falls: the first port's window until t0 falls; a gap until t1 falls, then the second port is selected and
    // its address settles until t2 falls; its window until t3 falls. All closed long before clk rises, so the flip-flops
    // changing can't reach an open latch.
    wire t0, t1, t2, t3;
    ram_delay write_turns (.clk(clk), .t0(t0), .t1(t1), .t2(t2), .t3(t3));
    wire second = !clk && !t1;
    wire we = (wen0 && !clk && t0) || (wen1 && !clk && !t2 && t3);
    wire [ABITS-1:0] waddr = second ? waddr1 : waddr0;
    wire [WIDTH-1:0] wdata = second ? wdata1 : wdata0;

    localparam OUT_BITS = RD_PORTS*WIDTH;
    wire [(BANKS+1)*OUT_BITS-1:0] bank_out;
    assign bank_out[OUT_BITS-1:0] = 0;
    assign RD_DATA = bank_out[BANKS*OUT_BITS +: OUT_BITS];
    genvar b, p;
    generate
        if (RD_PORTS == 4 && BANKS > 0) begin : map
            for (b = 0; b < BANKS; b = b + 1) begin : bank
                wire [RD_PORTS-1:0] enable;
                wire [RD_PORTS*ROW_BITS-1:0] row;
                for (p = 0; p < RD_PORTS; p = p + 1) begin : port
                    assign row[p*ROW_BITS +: ROW_BITS] = RD_ADDR[p*ABITS +: ROW_BITS];
                    if (BANK_BITS > 0)
                        assign enable[p] = RD_ADDR[p*ABITS+ROW_BITS +: (BANK_BITS > 0 ? BANK_BITS : 1)] == b;
                    else
                        assign enable[p] = 1'b1;
                end
                wire select = BANK_BITS > 0 ? waddr[ABITS-1:(BANK_BITS > 0 ? ROW_BITS : 0)] == b : 1'b1;
                wire [RD_PORTS*WIDTH-1:0] out;
                ram_4r1w #(.DEPTH(ROWS), .ABITS(ROW_BITS), .WIDTH(WIDTH), .INIT(INIT[b*ROWS*WIDTH +: ROWS*WIDTH])) array (
                    .aa(row[ROW_BITS-1:0]), .ab(row[2*ROW_BITS-1:ROW_BITS]),
                    .ac(row[3*ROW_BITS-1:2*ROW_BITS]), .ad(row[4*ROW_BITS-1:3*ROW_BITS]),
                    .ea(enable[0]), .eb(enable[1]), .ec(enable[2]), .ed(enable[3]),
                    .wa(waddr[ROW_BITS-1:0]), .we(we && select), .wd(wdata),
                    .ra(out[WIDTH-1:0]), .rb(out[2*WIDTH-1:WIDTH]), .rc(out[3*WIDTH-1:2*WIDTH]), .rd(out[4*WIDTH-1:3*WIDTH]));
                assign bank_out[(b+1)*OUT_BITS +: OUT_BITS] = bank_out[b*OUT_BITS +: OUT_BITS] | out;
            end
        end
    endgenerate
endmodule
