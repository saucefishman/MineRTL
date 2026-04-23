module main (
    input clk,
    output reg [3:0] count
);
    always @(posedge clk)
        count <= count + 4'd1;
endmodule
