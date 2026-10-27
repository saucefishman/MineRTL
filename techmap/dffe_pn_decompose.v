(* techmap_celltype = "$_DFFE_PN_" *)
module _DFFE_PN_ (input D, C, E, output Q);
    wire C_n;
    $_NOT_ inv (.A(C), .Y(C_n));
    $_DFFE_PP_ ff (.D(D), .C(C_n), .E(E), .Q(Q));
endmodule
