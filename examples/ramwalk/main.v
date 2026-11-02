// Walks the tiled RAM with --tiled-ram, which can't be preloaded: the first pass writes word n with bit n set, later passes
// only read. up shows a lit bit climbing through the words, down the same bit falling, on the second read port.
module main(input clk, output [2:0] count, output [7:0] up, output [7:0] down);

    reg [2:0] counter = 0;
    reg filled = 0;
    reg [7:0] data[0:7];

    assign count = counter;
    assign up = data[counter];
    assign down = data[~counter];

    always @(posedge clk) begin
        counter <= counter + 1;
        if (counter == 7) begin
            filled <= 1;
        end
        if (!filled) begin
            data[counter] <= 8'd1 << counter;
        end
    end

endmodule
