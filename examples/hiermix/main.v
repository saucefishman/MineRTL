module fa (input a, input b, input ci, output s, output co);
    assign s = a ^ b ^ ci;
    assign co = (a & b) | (ci & (a ^ b));
endmodule

module add4 (input [3:0] a, input [3:0] b, output [3:0] s, output co);
    wire [4:0] c;
    assign c[0] = 1'b0;
    fa f0 (.a(a[0]), .b(b[0]), .ci(c[0]), .s(s[0]), .co(c[1]));
    fa f1 (.a(a[1]), .b(b[1]), .ci(c[1]), .s(s[1]), .co(c[2]));
    fa f2 (.a(a[2]), .b(b[2]), .ci(c[2]), .s(s[2]), .co(c[3]));
    fa f3 (.a(a[3]), .b(b[3]), .ci(c[3]), .s(s[3]), .co(c[4]));
    assign co = c[4];
endmodule

module cnt4 (input clk, input en, output reg [3:0] q);
    always @(posedge clk) if (en) q <= q + 1;
endmodule

module main (input clk, input en_a, input en_b, output [3:0] sum, output co);
    wire [3:0] qa, qb;
    cnt4 ca (.clk(clk), .en(en_a), .q(qa));
    cnt4 cb (.clk(clk), .en(en_b), .q(qb));
    add4 ad (.a(qa), .b(qb), .s(sum), .co(co));
endmodule
