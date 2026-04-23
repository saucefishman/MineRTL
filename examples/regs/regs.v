module main(input clk,
    input raddr0, output [1:0]rdata0,
    input wen, input waddr, input [1:0]wdata);

    reg [1:0]data[0:1];

    assign rdata0 = data[raddr0];

    always @(posedge clk) begin
        if (wen) begin
            data[waddr] <= wdata;
        end
    end

endmodule
