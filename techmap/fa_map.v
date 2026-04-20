module \$add (A, B, Y);
  parameter A_SIGNED = 0;
  parameter B_SIGNED = 0;
  parameter A_WIDTH = 1;
  parameter B_WIDTH = 1;
  parameter Y_WIDTH = 1;

  input [A_WIDTH-1:0] A;
  input [B_WIDTH-1:0] B;
  output [Y_WIDTH-1:0] Y;

  wire cout;

  fulladder #(.WIDTH(Y_WIDTH)) fa (
    .A(A),
    .B(B),
    .Cin(1'b0),
    .S(Y),
    .Cout(cout)
  );
endmodule
