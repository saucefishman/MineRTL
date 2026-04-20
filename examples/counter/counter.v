module main (
    input clk,
    input increment,
    input reset,
    output reg [3:0] count
);
    always @(posedge clk)
        if (reset)
            count <= 4'd0;
        else if (increment)
            count <= count + 4'd1;
endmodule
