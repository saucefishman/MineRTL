// Reads memory out in order: q shows the word at count's low bits, count advances every clock.
// wen stores count into the current word, so the memory stays flip-flops rather than a ROM.
module main(input _clk, input switch, input wen, output [15:0] count, output [15:0] q);

    wire clk = _clk & switch;

    reg [15:0] counter = 0;
    reg [15:0] data[0:7];

    initial begin
        $readmemh("memread.hex", data);
    end

    assign count = counter;
    assign q = data[counter[2:0]];

    always @(posedge clk) begin
        counter <= counter + 1;
        if (wen) begin
            data[counter[2:0]] <= counter;
        end
    end

endmodule
