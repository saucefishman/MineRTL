// Clocks that rest high at power-up, with one input per flip-flop kind: settling must keep every initial value.
// The enabled pair has enable high at rest, so it would capture too if the clock weren't held. See scripts/dev/check_powerup.py.
module main(input _clk, input d, input hold, output [3:0] q);

    wire clk = ~_clk;

    reg plain_high = 1;
    reg plain_low = 0;
    reg enabled_high = 1;
    reg enabled_low = 0;

    assign q = {enabled_low, enabled_high, plain_low, plain_high};

    always @(posedge clk) begin
        plain_high <= d;
        plain_low <= ~d;
        if (~hold) begin
            enabled_high <= d;
            enabled_low <= ~d;
        end
    end

endmodule
