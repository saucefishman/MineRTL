module main(
    input clk,
    input a,
    output reg y
);
    always @(posedge clk) begin
        y <= y ^ a;
    end
endmodule