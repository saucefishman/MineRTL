// Paged sparse memory: full 15-bit word address space, 2 pages of 32 words.
// Pages are claimed on first write to their upper address bits.
// Unmapped reads return 0. Writes to a third page are dropped and full is high.
module mem(input clk,
    input [15:1]raddr0, output [15:0]rdata0,
    input [15:1]daddr, output [15:0]rdata1,
    input wen, input [15:0]wdata, output full);

    reg [9:0]tag0, tag1;
    reg valid0 = 0, valid1 = 0;

    reg [15:0]data[0:63];

    wire [9:0]page_r = raddr0[15:6];
    wire [9:0]page_d = daddr[15:6];

    wire r_hit0 = valid0 & (tag0 == page_r);
    wire r_hit1 = valid1 & (tag1 == page_r);
    wire d_hit0 = valid0 & (tag0 == page_d);
    wire d_hit1 = valid1 & (tag1 == page_d);

    wire miss = ~(d_hit0 | d_hit1);
    // on a miss, the next free page is 0 until page 0 is claimed
    wire wpage = miss ? valid0 : d_hit1;

    assign full = valid1;
    assign rdata0 = {16{r_hit0 | r_hit1}} & data[{r_hit1, raddr0[5:1]}];
    assign rdata1 = {16{d_hit0 | d_hit1}} & data[{d_hit1, daddr[5:1]}];

    always @(posedge clk) begin
        if (wen & ~(miss & valid1)) begin
            data[{wpage, daddr[5:1]}] <= wdata;
        end
        if (wen & miss & ~valid0) begin
            tag0 <= page_d;
            valid0 <= 1;
        end
        if (wen & miss & valid0 & ~valid1) begin
            tag1 <= page_d;
            valid1 <= 1;
        end
    end

endmodule
