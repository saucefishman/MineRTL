module \$_DLATCH_P_ (E, D, Q);
  input E, D;
  output Q;
  assign Q = E ? D : 1'b0;
endmodule
