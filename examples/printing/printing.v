// Prints "Hello!" and a lap count forever. Build with --debug-prints and run with
// mchprs_headless test --probes to see the output.
module main (
    input clk,
    output reg [2:0] index,
    output reg [3:0] lap
);
    initial index = 0;
    initial lap = 0;

    reg [7:0] char;
    always @(*) begin
        case (index)
            3'd0: char = "H";
            3'd1: char = "e";
            3'd2: char = "l";
            3'd3: char = "l";
            3'd4: char = "o";
            3'd5: char = "!";
            default: char = " ";
        endcase
    end

    always @(posedge clk) begin
        if (index == 3'd6) begin
            index <= 0;
            lap <= lap + 4'd1;
            $display(" lap %0d (0x%h)", lap, lap);
        end
        else begin
            index <= index + 3'd1;
            $write("%c", char);
        end
    end
endmodule
