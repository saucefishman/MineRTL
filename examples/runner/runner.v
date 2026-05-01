module main(
    input clk,
    output reg[3:0] col0,
    output reg[3:0] col1,
    output reg[3:0] col2,
    output reg[3:0] col3
);
    reg [3:0] count;

    always @(posedge clk) begin
        count <= count + 4'd1;
        col0 <= 4'b0;
      col1 <= 4'b0;
      col2 <= 4'b0;
      col3 <= 4'b0;
        col0[3 - count[1:0]] <= count[3:2] == 3 ? 1'b1 : 1'b0;
        col1[3 - count[1:0]] <= count[3:2] == 2 ? 1'b1 : 1'b0;
        col2[3 - count[1:0]] <= count[3:2] == 1 ? 1'b1 : 1'b0;
        col3[3 - count[1:0]] <= count[3:2] == 0 ? 1'b1 : 1'b0;
    end
endmodule