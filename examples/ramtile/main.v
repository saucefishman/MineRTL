// Smallest memory that takes the tiled array with --tiled-ram: 4 words of 2 bits, 2 reads, 1 write.
module main(input clk, input wen, input [1:0] waddr, input [1:0] wdata,
            input [1:0] raddr0, input [1:0] raddr1, output [1:0] rdata0, output [1:0] rdata1);

    reg [1:0] data[0:3];

    assign rdata0 = data[raddr0];
    assign rdata1 = data[raddr1];

    always @(posedge clk) begin
        if (wen) begin
            data[waddr] <= wdata;
        end
    end

endmodule
