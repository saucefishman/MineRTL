module main(input clk,
    input[1:0] raddr0, output [1:0]rdata0,
    input wen, input[1:0] waddr, input [1:0]wdata);

    reg [1:0]data[0:3];

    assign rdata0 = data[raddr0];

    always @(posedge clk) begin
        if (wen) begin
            data[waddr] <= wdata;
        end
    end

endmodule
