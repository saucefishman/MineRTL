// Small stand-in for video's output-pin extension routing: 8 shift registers whose
// bits each extend to a target on a 3-spaced grid in the z=0 output layer.
module main(
    input clk,
    input i0,
    input i1,
    input i2,
    input i3,
    input i4,
    input i5,
    input i6,
    input i7,
    output reg[5:0] o0,
    output reg[5:0] o1,
    output reg[5:0] o2,
    output reg[5:0] o3,
    output reg[5:0] o4,
    output reg[5:0] o5,
    output reg[5:0] o6,
    output reg[5:0] o7
);
    always @(posedge clk) begin
        o0 <= {o0[4:0], i0};
        o1 <= {o1[4:0], i1};
        o2 <= {o2[4:0], i2};
        o3 <= {o3[4:0], i3};
        o4 <= {o4[4:0], i4};
        o5 <= {o5[4:0], i5};
        o6 <= {o6[4:0], i6};
        o7 <= {o7[4:0], i7};
    end
endmodule
