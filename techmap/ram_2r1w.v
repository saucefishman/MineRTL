// Memories with exactly two asynchronous reads and one rising edge write become a tiled array (placement_engine/ram.py).
// Anything else fails this map and goes on to memory_map.
// TODO: other port counts, partial writes, preloaded contents. Move decoders into the array: row pins are 3 x depth.
(* techmap_celltype = "$mem_v2" *)
module _ram_2r1w (RD_CLK, RD_EN, RD_ARST, RD_SRST, RD_ADDR, RD_DATA, WR_CLK, WR_EN, WR_ADDR, WR_DATA);
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
    function automatic whole_word_write(input integer unused);
        integer i;
        begin
            whole_word_write = 1;
            for (i = 1; i < WIDTH; i = i + 1)
                if (_TECHMAP_CONNMAP_WR_EN_[i*CONN +: CONN] !== _TECHMAP_CONNMAP_WR_EN_[0 +: CONN])
                    whole_word_write = 0;
        end
    endfunction

    wire _TECHMAP_FAIL_ = RD_PORTS != 2 || WR_PORTS != 1 || RD_CLK_ENABLE !== 2'b00 || WR_CLK_ENABLE !== 1'b1
        || WR_CLK_POLARITY !== 1'b1 || OFFSET != 0 || SIZE != 1 << ABITS || RD_WIDE_CONTINUATION !== 2'b00
        || WR_WIDE_CONTINUATION !== 1'b0 || INIT !== {SIZE*WIDTH{1'bx}} || !whole_word_write(0);

    // Writes land just after the falling edge, from flip-flops that hold still until the next rising edge. The window
    // closes itself long before that edge, so the flip-flops changing can't reach an open latch. Reads see it by that edge.
    wire [ABITS-1:0] waddr;
    wire [WIDTH-1:0] wdata;
    wire wen;
    \$dff #(.WIDTH(ABITS), .CLK_POLARITY(1'b1)) waddr_reg (.CLK(WR_CLK), .D(WR_ADDR), .Q(waddr));
    \$dff #(.WIDTH(WIDTH), .CLK_POLARITY(1'b1)) wdata_reg (.CLK(WR_CLK), .D(WR_DATA), .Q(wdata));
    \$dff #(.WIDTH(1), .CLK_POLARITY(1'b1)) wen_reg (.CLK(WR_CLK), .D(WR_EN[0]), .Q(wen));

    wire [SIZE-1:0] nra, nrb, lock;
    wire window;
    ram_window window_gen (.clk(WR_CLK), .window(window));
    genvar k;
    generate
        if (RD_PORTS == 2) begin : map
            for (k = 0; k < SIZE; k = k + 1) begin : row
                assign nra[k] = RD_ADDR[ABITS-1:0] != k;
                assign nrb[k] = RD_ADDR[2*ABITS-1:ABITS] != k;
                assign lock[k] = !(wen && waddr == k && window);
            end
            ram_2r1w #(.DEPTH(SIZE), .WIDTH(WIDTH)) _TECHMAP_REPLACE_ (
                .nra(nra), .nrb(nrb), .lock(lock), .wd(wdata), .ra(RD_DATA[WIDTH-1:0]), .rb(RD_DATA[2*WIDTH-1:WIDTH]));
        end
    endgenerate
endmodule
