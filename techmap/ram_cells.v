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
