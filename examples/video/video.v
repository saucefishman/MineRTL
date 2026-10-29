module main(
    input clk,
    input i0,
    input i1,
    input i2,
    input i3,
    input i4,
    input i5,
    input i6,
    input i7,
    output reg[11:0] o0,
    output reg[11:0] o1,
    output reg[11:0] o2,
    output reg[11:0] o3,
    output reg[11:0] o4,
    output reg[11:0] o5,
    output reg[11:0] o6,
    output reg[11:0] o7,
    output reg[11:0] o8,
    output reg[11:0] o9,
    output reg[11:0] o10,
    output reg[11:0] o11,
    output reg[11:0] o12,
    output reg[11:0] o13,
    output reg[11:0] o14,
    output reg[11:0] o15,

);
    wire[7:0] i = {i7, i6, i5, i4, i3, i2, i1, i0};

    // 24 bytes: cnt=2k → bit k of o0..o7 from i[0..7], cnt=2k+1 → bit k of o8..o15
    reg[4:0] cnt;
    reg[11:0] frame[0:15];

    always @(posedge clk) begin
        if (cnt == 5'd0) begin
            o0  <= frame[0];
            o1  <= frame[1];
            o2  <= frame[2];
            o3  <= frame[3];
            o4  <= frame[4];
            o5  <= frame[5];
            o6  <= frame[6];
            o7  <= frame[7];
            o8  <= frame[8];
            o9  <= frame[9];
            o10 <= frame[10];
            o11 <= frame[11];
            o12 <= frame[12];
            o13 <= frame[13];
            o14 <= frame[14];
            o15 <= frame[15];
        end

        if (cnt[0] == 1'b0) begin
            frame[0][cnt[4:1]]  <= i[0];
            frame[1][cnt[4:1]]  <= i[1];
            frame[2][cnt[4:1]]  <= i[2];
            frame[3][cnt[4:1]]  <= i[3];
            frame[4][cnt[4:1]]  <= i[4];
            frame[5][cnt[4:1]]  <= i[5];
            frame[6][cnt[4:1]]  <= i[6];
            frame[7][cnt[4:1]]  <= i[7];
        end else begin
            frame[8][cnt[4:1]]  <= i[0];
            frame[9][cnt[4:1]]  <= i[1];
            frame[10][cnt[4:1]] <= i[2];
            frame[11][cnt[4:1]] <= i[3];
            frame[12][cnt[4:1]] <= i[4];
            frame[13][cnt[4:1]] <= i[5];
            frame[14][cnt[4:1]] <= i[6];
            frame[15][cnt[4:1]] <= i[7];
        end

        cnt <= (cnt == 5'd23) ? 5'd0 : cnt + 5'd1;
    end
endmodule
