module \$_SDFF_PN0_ (C, R, D, Q);
  input C, R, D;
  output Q;
  wire C_n, D_mux;
  \$_NOT_ inv (.A(C), .Y(C_n));
  \$_MUX_ mux (.A(D), .B(1'b0), .S(R), .Y(D_mux));
  \$_DFF_P_ ff (.C(C_n), .D(D_mux), .Q(Q));
endmodule

module \$_SDFF_PN1_ (C, R, D, Q);
  input C, R, D;
  output Q;
  wire C_n, D_mux;
  \$_NOT_ inv (.A(C), .Y(C_n));
  \$_MUX_ mux (.A(D), .B(1'b1), .S(R), .Y(D_mux));
  \$_DFF_P_ ff (.C(C_n), .D(D_mux), .Q(Q));
endmodule

module \$_SDFF_PP0_ (C, R, D, Q);
  input C, R, D;
  output Q;
  wire D_mux;
  \$_MUX_ mux (.A(D), .B(1'b0), .S(R), .Y(D_mux));
  \$_DFF_P_ ff (.C(C), .D(D_mux), .Q(Q));
endmodule

module \$_SDFF_PP1_ (C, R, D, Q);
  input C, R, D;
  output Q;
  wire D_mux;
  \$_MUX_ mux (.A(D), .B(1'b1), .S(R), .Y(D_mux));
  \$_DFF_P_ ff (.C(C), .D(D_mux), .Q(Q));
endmodule

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
