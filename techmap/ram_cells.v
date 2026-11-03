// Port directions of the tiled array, for yosys; the blocks come from placement_engine/ram.py.
(* blackbox *)
module ram_2r1w #(parameter DEPTH = 2, parameter ABITS = 1, parameter WIDTH = 1) (
    input [ABITS-1:0] aa, ab, wa,
    input we,
    input [WIDTH-1:0] wd,
    output [WIDTH-1:0] ra, rb);
endmodule

// High for a fixed time after clk falls. A block, because opt cancels inverter pairs in a delay made of gates.
(* blackbox *)
module ram_window (input clk, output window);
endmodule

(* blackbox *)
module ram_stack #(parameter ROW_BITS = 1, parameter BANKS = 1, parameter BASE = 0, parameter ABITS = 1, parameter WIDTH = 1,
                   parameter INIT = 0) (
    input [ABITS-1:0] aa, ab, ac, ad, wa,
    input we,
    input [WIDTH-1:0] wd,
    output [WIDTH-1:0] ra, rb, rc, rd);
endmodule

// The clock late by four amounts, for the write turns of ram_4r2w.
(* blackbox *)
module ram_delay (input clk, output t0, t1, t2, t3);
endmodule
