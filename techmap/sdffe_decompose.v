module \$sdffe (CLK, SRST, EN, D, Q);
  parameter WIDTH = 1;
  parameter CLK_POLARITY = 1;
  parameter SRST_POLARITY = 1;
  parameter SRST_VALUE = 1'bx;
  parameter EN_POLARITY = 1;

  input CLK, SRST, EN;
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

  \$dffe #(.WIDTH(WIDTH), .CLK_POLARITY(CLK_POLARITY), .EN_POLARITY(EN_POLARITY)) ff (
    .CLK(CLK),
    .EN(EN),
    .D(D_in),
    .Q(Q)
  );
endmodule
