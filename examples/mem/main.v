module main(input clk,
    input [15:1]raddr0, output [15:0]rdata0,
    input [15:1]daddr, output [15:0]rdata1,
    input wen, input [15:0]wdata, output full);

    mem mem(clk,
        raddr0, rdata0,
        daddr, rdata1,
        wen, wdata, full);

endmodule
