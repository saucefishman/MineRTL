// Port directions of the tiled array, for yosys; the blocks come from placement_engine/ram.py.
(* blackbox *)
module ram_2r1w #(parameter DEPTH = 1, parameter WIDTH = 1) (
    input [DEPTH-1:0] nra, nrb, lock,
    input [WIDTH-1:0] wd,
    output [WIDTH-1:0] ra, rb);
endmodule

// High for a fixed time after clk falls. A block, because opt cancels inverter pairs in a delay made of gates.
(* blackbox *)
module ram_window (input clk, output window);
endmodule
