module main(
    input[3:0] a,
    input b,
    output[3:0] y
);
    assign y = a ^ {4{b}};
endmodule