// A full adder with a tied input is a half adder or less: plain gates are a quarter of its tiles and
// a third of its delay, and ABC can then fold them. Adders with three live inputs stay as they are.
module fulladder (A, B, Cin, S, Cout);
  parameter _TECHMAP_CONSTMSK_A_ = 0;
  parameter _TECHMAP_CONSTMSK_B_ = 0;
  parameter _TECHMAP_CONSTMSK_Cin_ = 0;
  input A, B, Cin;
  output S, Cout;
  generate
    if (_TECHMAP_CONSTMSK_A_ || _TECHMAP_CONSTMSK_B_ || _TECHMAP_CONSTMSK_Cin_) begin
      assign S = A ^ B ^ Cin;
      assign Cout = (A & B) | (Cin & (A ^ B));
    end else begin
      wire _TECHMAP_FAIL_ = 1;
    end
  endgenerate
endmodule
