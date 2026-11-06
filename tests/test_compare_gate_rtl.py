import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from compare_gate_rtl import Netlist  # noqa: E402


def test_netlist_reads_ram_rom(tmp_path):
    # 2 address bits, 4-bit words 3, 5, 9 (row 3 past INIT reads 0); INIT is written msb first.
    words = [3, 5, 9]
    init = "".join(format(w, "04b") for w in reversed(words))
    param = lambda n: format(n, "032b")
    cells = {
        "rom": {"type": "ram_rom", "parameters": {"ABITS": param(2), "DEPTH": param(4), "WIDTH": param(4), "INIT": init},
                "connections": {"aa": [2, 3], "ra": [4, 5, 6, 7]}},
    }
    path = tmp_path / "netlist.json"
    path.write_text(json.dumps({"modules": {"main": {"cells": cells, "ports": {"clk": {"bits": [1]}, "a0": {"bits": [2]},
                                                                                "a1": {"bits": [3]}}}}}))
    sim = Netlist(path)
    for address, word in enumerate(words + [0]):
        sim.set_input("a0", address & 1)
        sim.set_input("a1", address >> 1)
        assert [sim.values[b] for b in (4, 5, 6, 7)] == [word >> i & 1 for i in range(4)]
