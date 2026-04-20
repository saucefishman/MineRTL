module \$sdff (CLK, SRST, D, Q);
  parameter WIDTH = 1;
  parameter CLK_POLARITY = 1;
  parameter SRST_POLARITY = 1;
  parameter SRST_VALUE = 1'bx;

  input CLK, SRST;
  input [WIDTH-1:0] D;
  output [WIDTH-1:0] Q;

  wire srst_active = SRST_POLARITY ? SRST : !SRST;
  wire [WIDTH-1:0] D_in;

  \$mux #(.WIDTH(WIDTH)) mux (
    .A(D),
    .B(SRST_VALUE),
    .S(srst_active),
    .Y(D_in)
  );

  \$dff #(.WIDTH(WIDTH), .CLK_POLARITY(CLK_POLARITY)) ff (
    .CLK(CLK),
    .D(D_in),
    .Q(Q)
  );
endmodule
