// A small single-cycle CPU: 8-bit data, four registers, a carry flag, and 16 words of program ROM.
//
// Every instruction is 16 bits: op[15:12] rd[11:10] rs[9:8] imm[7:0].
//   0 LI   rd, imm   rd = imm
//   1 ADD  rd, rs    rd = rd + rs, C = carry out
//   2 SUB  rd, rs    rd = rd - rs, C = borrow
//   3 AND  rd, rs    rd = rd & rs
//   4 OR   rd, rs    rd = rd | rs
//   5 XOR  rd, rs    rd = rd ^ rs
//   6 MOV  rd, rs    rd = rs
//   7 ADDI rd, imm   rd = rd + imm, C = carry out
//   8 OUT  rs        out = rs
//   9 JMP  imm       pc = imm
//   A JC   imm       pc = imm if C is set
//   B JZ   rs, imm   pc = imm if rs is zero
// Other opcodes do nothing. The program (cpu.hex) prints Fibonacci numbers on the out lamps and starts over when the
// next one would overflow.
module main(input clk, output [7:0] out, output [3:0] pc_out);

    reg [15:0] rom[0:15];
    initial begin
        $readmemh("cpu.hex", rom);
    end

    reg [3:0] pc = 0;
    reg [7:0] regs[0:3];
    reg [7:0] out_reg = 0;
    reg carry = 0;

    wire [15:0] instr = rom[pc];
    wire [3:0] op = instr[15:12];
    wire [1:0] rd = instr[11:10];
    wire [1:0] rs = instr[9:8];
    wire [7:0] imm = instr[7:0];

    wire [7:0] a = regs[rd];
    wire [7:0] b = regs[rs];
    wire [7:0] operand = op == 4'h7 ? imm : b;
    wire [8:0] sum = op == 4'h2 ? {1'b0, a} - {1'b0, b} : {1'b0, a} + {1'b0, operand};

    reg [7:0] result;
    always @(*) begin
        case (op)
            4'h0: result = imm;
            4'h3: result = a & b;
            4'h4: result = a | b;
            4'h5: result = a ^ b;
            4'h6: result = b;
            default: result = sum[7:0];
        endcase
    end

    wire writes = op <= 4'h7;
    wire sets_carry = op == 4'h1 || op == 4'h2 || op == 4'h7;
    wire jump = op == 4'h9 || (op == 4'hA && carry) || (op == 4'hB && b == 0);

    assign out = out_reg;
    assign pc_out = pc;

    always @(posedge clk) begin
        pc <= jump ? imm[3:0] : pc + 1;
        if (writes) regs[rd] <= result;
        if (sets_carry) carry <= sum[8];
        if (op == 4'h8) out_reg <= b;
    end

endmodule
