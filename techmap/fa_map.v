module \$add (A, B, Y);
  parameter A_SIGNED = 0;
  parameter B_SIGNED = 0;
  parameter A_WIDTH = 1;
  parameter B_WIDTH = 1;
  parameter Y_WIDTH = 1;

  input [A_WIDTH-1:0] A;
  input [B_WIDTH-1:0] B;
  output [Y_WIDTH-1:0] Y;

  wire [Y_WIDTH:0] carry;
  assign carry[0] = 1'b0;

  genvar i;
  generate
    for (i = 0; i < Y_WIDTH; i = i + 1) begin : fa_bit
      wire a_i = (i < A_WIDTH) ? A[i] : 1'b0;
      wire b_i = (i < B_WIDTH) ? B[i] : 1'b0;
      fulladder fa_i (
        .A(a_i),
        .B(b_i),
        .Cin(carry[i]),
        .S(Y[i]),
        .Cout(carry[i+1])
      );
    end
  endgenerate
endmodule
