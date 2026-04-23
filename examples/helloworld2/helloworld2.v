module main(
    input a,
    input b,
    input c,
    input d,
    output y,
    output z
);
    assign y = a ^ b;
    assign z = c & d;
endmodule
