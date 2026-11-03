// Two preloaded memories on shared addresses: a RAM group whose contents must read back before any write.
// Build with --tiled-ram, check with scripts/check_preload.py.
module main(input clk, input [3:0] addr0, input [3:0] addr1, input [3:0] addr2, input [3:0] addr3,
    output [1:0] a0, output [1:0] a1, output [1:0] a2, output [1:0] a3,
    output [2:0] b0, output [2:0] b1, output [2:0] b2, output [2:0] b3,
    input wen0, input [3:0] waddr0, input [1:0] wdata0, input wen1, input [3:0] waddr1, input [1:0] wdata1);
    reg [1:0] m[0:15];
    reg [2:0] n[0:15];
    integer i;
    initial for (i = 0; i < 16; i = i + 1) begin
        m[i] = (i * 7 + 1) % 4;
        n[i] = (i * 5 + 3) % 8;
    end
    assign a0 = m[addr0]; assign a1 = m[addr1]; assign a2 = m[addr2]; assign a3 = m[addr3];
    assign b0 = n[addr0]; assign b1 = n[addr1]; assign b2 = n[addr2]; assign b3 = n[addr3];
    always @(posedge clk) begin
        if (wen0) m[waddr0] <= wdata0;
        if (wen1) m[waddr1] <= wdata1;
        if (wen0) n[waddr0] <= {wdata0[0], wdata0};
        if (wen1) n[waddr1] <= {wdata1[0], wdata1};
    end
endmodule
