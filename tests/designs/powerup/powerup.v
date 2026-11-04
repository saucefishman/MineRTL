// Smallest circuit found that loses state at power-up when pasted unsettled: on --place-seed 7, a false
// clock pulse and a false enable pulse overlap on one word, which overwrites its 1. See scripts/dev/check_powerup.py.
module main(input _clk, input switch, input wen, output q);

    wire clk = _clk & switch;

    reg [1:0] counter = 0;
    reg data[0:3];

    initial begin
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
        data[3] = 1;
    end

    assign q = data[counter];

    always @(posedge clk) begin
        counter <= counter + 1;
        if (wen) begin
            data[counter] <= 0;
        end
    end

endmodule
