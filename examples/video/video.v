// A 16x12 screen playing 128 frames of video from one preloaded memory: each word is a frame, and a counter steps
// through them, a frame per clock. Each output is a column of the screen, dark0 at the left, and its bit k is 1 where
// the pixel k rows from the bottom is dark: pin_targets.json lays each pixel out as a 3x3 tile of lamps, lit while its
// pin is low. make_frames.py writes video.hex and pin_targets.json.
module main(input clk, output [11:0] dark0, dark1, dark2, dark3, dark4, dark5, dark6, dark7,
                                     dark8, dark9, dark10, dark11, dark12, dark13, dark14, dark15);

    reg [191:0] frames[0:127];

    initial begin
        $readmemh("video.hex", frames);
    end

    reg [6:0] frame = 0;

    always @(posedge clk) begin
        frame <= frame + 1;
    end

    // dark0 in the top bits: the memory's bit 0 is at its west end, as the screen's right column is.
    assign {dark0, dark1, dark2, dark3, dark4, dark5, dark6, dark7,
            dark8, dark9, dark10, dark11, dark12, dark13, dark14, dark15} = frames[frame];

endmodule
