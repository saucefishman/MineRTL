module main(
    input clk,
    input i0,
    input i1,
    input i2,
    input i3,
    output reg[11:0] o0,
    output reg[11:0] o1,
    output reg[11:0] o2,
    output reg[11:0] o3,
    output reg[11:0] o4,
    output reg[11:0] o5,
    output reg[11:0] o6,
    output reg[11:0] o7,
);
    wire[3:0] i = {i3, i2, i1, i0};

    // Half of video: 24 nibbles: cnt=2k → bit k of o0..o3 from i[0..3], cnt=2k+1 → bit k of o4..o7
    reg[4:0] cnt;
    reg[11:0] frame[0:7];

    always @(posedge clk) begin
        if (cnt == 5'd0) begin
            o0 <= frame[0];
            o1 <= frame[1];
            o2 <= frame[2];
            o3 <= frame[3];
            o4 <= frame[4];
            o5 <= frame[5];
            o6 <= frame[6];
            o7 <= frame[7];
        end

        if (cnt[0] == 1'b0) begin
            frame[0][cnt[4:1]] <= i[0];
            frame[1][cnt[4:1]] <= i[1];
            frame[2][cnt[4:1]] <= i[2];
            frame[3][cnt[4:1]] <= i[3];
        end else begin
            frame[4][cnt[4:1]] <= i[0];
            frame[5][cnt[4:1]] <= i[1];
            frame[6][cnt[4:1]] <= i[2];
            frame[7][cnt[4:1]] <= i[3];
        end

        cnt <= (cnt == 5'd23) ? 5'd0 : cnt + 5'd1;
    end
endmodule
