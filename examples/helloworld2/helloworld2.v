module main(
    input a,
    input b,
    output y,
    output z
);
    assign y = a ^ b;
    assign z = a & b;
endmodule
