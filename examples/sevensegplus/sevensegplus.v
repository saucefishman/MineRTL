module main(
    input clk,
    output reg [6:0] seg0,
    output reg [6:0] seg1
);
    reg [3:0] count0;
//    reg [3:0] count1;

    always @(posedge clk) begin
        if (count0 >= 4'd9) begin
            count0 <= 4'd0;
//            if (count1 >= 4'd9) begin
//                count1 <= 4'd0;
//            end
//            else begin
//                count1 <= count1 + 4'd1;
//            end
        end
        else begin
            count0 <= count0 + 4'd1;
        end
    end

    // seg[6:0] = abcdefg, common cathode active high
    always @(*) begin
        case (count0)
            4'd0: seg0 = 7'b1111110;
            4'd1: seg0 = 7'b0110000;
            4'd2: seg0 = 7'b1101101;
            4'd3: seg0 = 7'b1111001;
            4'd4: seg0 = 7'b0110011;
            4'd5: seg0 = 7'b1011011;
            4'd6: seg0 = 7'b1011111;
            4'd7: seg0 = 7'b1110000;
            4'd8: seg0 = 7'b1111111;
            4'd9: seg0 = 7'b1111011;
            default: seg0 = 7'b0000000;
        endcase
        case (count0)
//            4'd0: seg1 = 7'b1111110;
//            4'd1: seg1 = 7'b0110000;
//            4'd2: seg1 = 7'b1101101;
//            4'd3: seg1 = 7'b1111001;
//            4'd4: seg1 = 7'b0110011;
//            4'd5: seg1 = 7'b1011011;
//            4'd6: seg1 = 7'b1011111;
//            4'd7: seg1 = 7'b1110000;
//            4'd8: seg1 = 7'b1111111;
//            4'd9: seg1 = 7'b1111011;
            4'd10: seg1 = 7'b1111110;
            4'd11: seg1 = 7'b0110000;
            4'd12: seg1 = 7'b1101101;
            4'd13: seg1 = 7'b1111001;
            4'd14: seg1 = 7'b0110011;
            4'd15: seg1 = 7'b1011011;
            default: seg1 = 7'b0000000;
        endcase
    end
endmodule
