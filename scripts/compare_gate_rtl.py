"""Compare a built schematic, net by net and cycle by cycle, against simulations of its netlist and of its RTL.

Reads every net's source and sink wire cells in mchprs_headless over a clocked run, then checks:
- local: each net's sinks equal its source; each gate's output follows its measured inputs; each flip-flop's Q took
  its measured D at the edge. Needs no reference and doesn't get lost after the first wrong value.
- netlist: each net against a two-state simulation of netlist.json that starts from the flip-flops' measured power-up.
- rtl (with --rtl): named nets against iverilog at the end of each low phase, X in the RTL ignored; checks the netlist
  simulation as well.

Usage: compare_gate_rtl.py <build dir> [--litematic F] [--pins F] [--half 3000] [--clocks 40] [--rtl tb_top file.v ...]
The build dir holds netlist.json, component_list.json, component_layout.json, out.litematic and pins.
Report goes to <build dir or --out>/compare.json. Needs mchprs_headless. Terminal cells assume a workspace at 0,0,0.
"""
import argparse
import gzip
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import nbtlib
import numpy as np
from litemapy import BlockState, Region
from minecraft_v.placement_engine.ram import rom_contents
from minecraft_v.placement_engine.verify import find_live_pin_wire
from settle_schematic import headless_binary

READABLE = ("minecraft:redstone_wire", "minecraft:repeater", "minecraft:comparator", "minecraft:redstone_torch",
            "minecraft:redstone_wall_torch", "minecraft:redstone_lamp")
Bit = int | str


def pin_terminal(comp: dict, pin: dict) -> tuple[int, int, int]:
    """The wire cell at a pin, as the builder's pin_terminal finds it: layout JSON keeps no side, so go by pin name."""
    ox, oy, oz = comp["origin"]
    px, py, pz = pin["offset"]
    x, y, z = ox + px, oy + py, oz + comp["footprint"]["depth"] - 1 - pz
    kind, name = comp["type"], pin["name"]
    if kind in ("INPUT_PIN", "OUTPUT_PIN"):
        return x, y, z
    if name in ("C", "E") or (kind == "MUX" and name == "S"):  # on the side, x says which
        return (x + 1 if px > 0 else x - 1), y, z
    if kind == "CUSTOM":
        north = "write_turns" in comp["id"] and name != "clk"
    else:
        north = pin["direction"] == "OUT"
    return x, y, z - 1 if north else z + 1


def load_region(path: Path) -> Region:
    """A litematic's main region, much faster than litemapy: its block array decoded with numpy."""
    with gzip.open(path) as f:
        region = nbtlib.File.parse(io.BytesIO(f.read()))["Regions"]["main"]
    w, h, l = (abs(int(region["Size"][k])) for k in "xyz")
    palette = []
    for entry in region["BlockStatePalette"]:
        block = BlockState(str(entry["Name"]))
        if "Properties" in entry:
            block = block.with_properties(**{str(k): str(v) for k, v in entry["Properties"].items()})
        palette.append(block)
    bits = max(math.ceil(math.log2(len(palette))), 2)
    longs = np.frombuffer(np.array(region["BlockStates"], dtype=np.int64).tobytes(), dtype=np.uint64)
    start = np.arange(w * h * l, dtype=np.uint64) * np.uint64(bits)
    word, offset = (start >> np.uint64(6)).astype(np.int64), start & np.uint64(63)
    mask = np.uint64((1 << bits) - 1)
    values = (longs[word] >> offset) & mask
    spill = offset + np.uint64(bits) > np.uint64(64)
    values[spill] |= (longs[word[spill] + 1] << (np.uint64(64) - offset[spill])) & mask
    result = Region(0, 0, 0, w, h, l)
    result._Region__palette = palette
    result._Region__blocks = values.astype(np.uint32).reshape((h, l, w)).transpose(2, 0, 1)
    return result


class Netlist:
    """Two-state simulation of a yosys JSON module, RAM blocks modelled per techmap/ram_4r2w.v."""

    def __init__(self, path: Path, top: str = "main"):
        module = json.load(open(path))["modules"][top]
        self.module = module
        self.cells = module["cells"]
        self.values: dict[Bit, int] = defaultdict(int, {"0": 0, "1": 1, "x": 0, "z": 0})
        self.flops = {n: c["connections"] for n, c in self.cells.items() if c["type"] in ("$_DFF_P_", "$_DFFE_PP_")}
        self.delays = [c["connections"] for c in self.cells.values() if c["type"] == "ram_delay"]
        self.rams = []
        for cell in self.cells.values():
            if cell["type"] == "ram_stack":
                self.rams.append(self._ram(cell))
        self.roms = [self._rom(c) for c in self.cells.values() if c["type"] == "ram_rom"]
        self.comb = self._levelize()
        self.clk = module["ports"]["clk"]["bits"][0]
        self.turns = 1  # t0..t3 high
        self.settle_all()

    @staticmethod
    def _ram(cell: dict) -> dict:
        p = {k: v for k, v in cell["parameters"].items()}
        width, row_bits, banks, base = (int(p[k], 2) for k in ("WIDTH", "ROW_BITS", "BANKS", "BASE"))
        init = p["INIT"][::-1]  # strings are MSB first
        mem = {}
        for i in range(banks << row_bits):
            word = init[i * width:(i + 1) * width]
            mem[i] = sum(1 << b for b, ch in enumerate(word) if ch == "1")
        return dict(conn=cell["connections"], width=width, row_bits=row_bits, banks=banks, base=base, mem=mem)

    @staticmethod
    def _rom(cell: dict) -> dict:
        """A ram_rom (techmap/ram_rom.v): one asynchronous read of its INIT words, rows past INIT read 0."""
        depth, width, words = rom_contents(cell["parameters"])
        return dict(conn=cell["connections"], words=words)

    def _levelize(self) -> list:
        kinds = {"$_AND_": ("A", "B"), "$_OR_": ("A", "B"), "$_XOR_": ("A", "B"), "$_NOT_": ("A",), "$_MUX_": ("A", "B", "S"),
                 "fulladder": ("A", "B", "Cin")}
        nodes = []
        driver = {}
        for name, cell in self.cells.items():
            kind = cell["type"]
            conn = cell["connections"]
            if kind in kinds:
                inputs = [conn[p][0] for p in kinds[kind]]
                outputs = [conn["S"][0], conn["Cout"][0]] if kind == "fulladder" else [conn["Y"][0]]
                nodes.append((kind, conn, inputs, outputs))
        for ram in self.rams:
            conn = ram["conn"]
            for port in "abcd":  # one port's address can come from another's data
                nodes.append(("ram", (ram, port), conn["a" + port], conn["r" + port]))
        for rom in self.roms:
            nodes.append(("rom", (rom,), rom["conn"]["aa"], rom["conn"]["ra"]))
        for i, node in enumerate(nodes):
            for bit in node[3]:
                driver[bit] = i
        users = defaultdict(list)
        waiting = [0] * len(nodes)
        for i, node in enumerate(nodes):
            for bit in set(node[2]):
                if bit in driver:
                    users[driver[bit]].append(i)
                    waiting[i] += 1
        ready = deque(i for i, n in enumerate(waiting) if n == 0)
        order = []
        while ready:
            i = ready.popleft()
            order.append(nodes[i])
            for j in users[i]:
                waiting[j] -= 1
                if waiting[j] == 0:
                    ready.append(j)
        if len(order) != len(nodes):
            raise ValueError(f"combinational loop through {len(nodes) - len(order)} cells")
        return order

    def settle_all(self) -> None:
        v = self.values
        for kind, conn, inputs, outputs in self.comb:
            if kind == "$_AND_":
                v[outputs[0]] = v[inputs[0]] & v[inputs[1]]
            elif kind == "$_OR_":
                v[outputs[0]] = v[inputs[0]] | v[inputs[1]]
            elif kind == "$_XOR_":
                v[outputs[0]] = v[inputs[0]] ^ v[inputs[1]]
            elif kind == "$_NOT_":
                v[outputs[0]] = 1 - v[inputs[0]]
            elif kind == "$_MUX_":
                v[outputs[0]] = v[inputs[1]] if v[inputs[2]] else v[inputs[0]]
            elif kind == "fulladder":
                total = v[inputs[0]] + v[inputs[1]] + v[inputs[2]]
                v[conn["S"][0]], v[conn["Cout"][0]] = total & 1, total >> 1
            elif kind == "rom":
                rom = conn[0]
                address = sum(v[b] << i for i, b in enumerate(rom["conn"]["aa"]))
                word = rom["words"][address] if address < len(rom["words"]) else 0
                for i, bit in enumerate(rom["conn"]["ra"]):
                    v[bit] = word >> i & 1
            else:
                self._read(*conn)

    def _read(self, ram: dict, port: str) -> None:
        conn, v = ram["conn"], self.values
        word = self._word(ram, sum(v[b] << i for i, b in enumerate(conn["a" + port])))
        for i, bit in enumerate(conn["r" + port]):
            v[bit] = word >> i & 1

    @staticmethod
    def _word(ram: dict, address: int) -> int:
        return ram["mem"].get(address - (ram["base"] << ram["row_bits"]), 0)

    def _write_rams(self) -> None:
        v = self.values
        for ram in self.rams:
            conn = ram["conn"]
            if v[conn["we"][0]]:
                offset = sum(v[b] << i for i, b in enumerate(conn["wa"])) - (ram["base"] << ram["row_bits"])
                if offset in ram["mem"]:
                    ram["mem"][offset] = sum(v[b] << i for i, b in enumerate(conn["wd"]))

    def _set_turns(self, mask: int) -> None:
        for conn in self.delays:
            for i in range(4):
                self.values[conn[f"t{i}"][0]] = mask >> i & 1

    def set_input(self, name: str, value: int) -> None:
        self.values[self.module["ports"][name]["bits"][0]] = value
        self.settle_all()

    def rise(self) -> None:
        """Clock up: flip-flops take what they saw before it."""
        before = {n: (self.values[c["D"][0]], self.values[c["E"][0]] if "E" in c else 1) for n, c in self.flops.items()}
        old_clock = {n: self.values[c["C"][0]] for n, c in self.flops.items()}
        self.values[self.clk] = 1
        self.settle_all()
        for name, conn in self.flops.items():
            if self.values[conn["C"][0]] and not old_clock[name] and before[name][1]:
                self.values[conn["Q"][0]] = before[name][0]
        self.settle_all()
        self._set_turns(0b1111)
        self.settle_all()

    def fall(self) -> None:
        """Clock down: the write turns of ram_4r2w, one after another."""
        self.values[self.clk] = 0
        self.settle_all()
        self._write_rams()
        self.settle_all()
        for mask in (0b1110, 0b1100, 0b1000, 0b0000):
            self._set_turns(mask)
            self.settle_all()
            self._write_rams()
            self.settle_all()


def live_pins_on_wire(build: Path, region: Region, terminals: dict, nets: dict) -> list:
    """Pins that drive their cell, with no net, and something conductive there: a short no net's samples can show."""
    taken = {terminals[key] for source, sinks in nets.values() for key in (source, *sinks) if key in terminals}
    live = {f"{comp['id']}:{pin['name']}": terminals[comp["id"], pin["name"]]
            for comp in json.load(open(build / "component_layout.json")) if comp["type"] in ("CUSTOM", "INPUT_PIN")
            for pin in comp["pins"] if pin["direction"] == "OUT" and (comp["id"], pin["name"]) in terminals
            and terminals[comp["id"], pin["name"]] not in taken}
    return find_live_pin_wire(region, live)


def endpoints(build: Path) -> tuple[dict, dict]:
    """(component, pin) -> terminal cell, and net id -> (source endpoint, sink endpoints)."""
    layout = {c["id"]: c for c in json.load(open(build / "component_layout.json"))}
    terminals = {}
    for comp in layout.values():
        for pin in comp["pins"]:
            terminals[comp["id"], pin["name"]] = pin_terminal(comp, pin)
    if (build / "terminals.json").exists():  # exact cells, from the build itself: replaces what the layout implies
        terminals = {(cid, pin): tuple(pos) for cid, pin, pos in json.load(open(build / "terminals.json"))}
    nets = {}
    for net in json.load(open(build / "component_list.json"))["nets"]:
        ends = [net["source"], *net["sinks"]]
        keys = []
        for e in ends:
            key = (e["component_id"], e["pin_name"])
            if key not in terminals and (e["pin_name"], e["pin_name"]) in terminals:
                key = (e["pin_name"], e["pin_name"])  # multi-bit IO splits into one component per bit
            keys.append(key)
        nets[net["net_id"]] = (keys[0], keys[1:])
    return terminals, nets


def run_headless(litematic: Path, pins_in: Path, cells: dict[str, tuple], half: int, clocks: int, work: Path,
                 mode: str, reuse: bool = False) -> list[tuple[str, dict[str, int]]]:
    """Samples at settled points: power-up, then every clock's high and low phases. reuse: read the last run's output."""
    pins = work / "pins.txt"
    with open(pins, "w") as f:
        f.write(open(pins_in).read())
        for name, (x, y, z) in cells.items():
            f.write(f"{name} {x} {y} {z} 0 0\n")
    script = ["set rst 1", f"settle {half}", "print init", "set clk 1", f"settle {half}", "print rh", "set clk 0",
              f"settle {half}", "print rl", "set rst 0", f"settle {half}", "print r0"]
    for i in range(clocks):
        script += ["set clk 1", f"settle {half}", f"print h{i}", "set clk 0", f"settle {half}", f"print l{i}"]
    path = work / "script.txt"
    path.write_text("\n".join(script) + "\n")
    out = work / "headless.txt"
    with open(os.devnull if reuse else out, "w") as f:
        if not reuse:
            subprocess.run([headless_binary(), "test", litematic, pins, path, "--modes", mode, "--add-lever", "clk", "--add-lever", "rst"],
                           stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False)
    samples = []
    for line in open(out):
        match = re.match(r"\s*\[\w+\] (\S+)\s+(.*)", line)
        if match and "=" in match[2]:
            samples.append((match[1], {k: int(v) for k, v in (kv.split("=") for kv in match[2].split())}))
    if not samples:
        raise RuntimeError(f"no samples from mchprs_headless, see {out}")
    return samples


def gate_output(kind: str, m: dict[str, int]) -> dict[str, int]:
    if kind == "AND":
        return {"Y": m["A"] & m["B"]}
    if kind == "OR":
        return {"Y": m["A"] | m["B"]}
    if kind == "XOR":
        return {"Y": m["A"] ^ m["B"]}
    if kind == "NOT":
        return {"Y": 1 - m["A"]}
    if kind == "MUX":
        return {"Y": m["B"] if m["S"] else m["A"]}
    if kind == "FULLADDER":
        total = m.get("A", 0) + m.get("B", 0) + m.get("Cin", 0)
        return {"S": total & 1, "Cout": total >> 1}
    return {}


def bit_of_net(netlist: Netlist, nets: dict) -> dict[str, Bit]:
    """Net id -> netlist bit, through the gate pins both sides name."""
    bits = {}
    for net_id, (source, sinks) in nets.items():
        for cid, pin in (source, *sinks):
            cell = netlist.cells.get(cid)
            if cell and pin in cell["connections"]:
                bits[net_id] = cell["connections"][pin][0]
                break
        else:
            match = re.fullmatch(r"net_(\d+)", net_id)
            if match:
                bits[net_id] = int(match[1])
    return bits


def rtl_samples(args, work: Path, deposits: dict[str, str] | None = None, regs: dict[str, int] | None = None,
                half_ns: int = 5) -> dict[str, list[str]]:
    """Named RTL signals after every phase, in the same order as the headless samples. deposits: reg -> value at time 0."""
    top, *files = args.rtl
    tb = work / "cmp_tb.v"
    samples = ["init", "rh", "rl", "r0"] + [f"{p}{i}" for i in range(args.clocks) for p in "hl"]
    start = "".join(f" dut.{name} = {len(value)}'b{value};\n" for name, value in (deposits or {}).items())
    tb.write_text(f"""`timescale 1ns/1ns
module cmp_tb; reg clk=0, rst=1; wire e0,e1,halt; wire [7:0] d0,d1;
{top} dut(clk,rst,e0,d0,e1,d1,halt);
initial begin
{start} $dumpfile("{work}/rtl.vcd"); $dumpvars(0,dut);
 #{half_ns} clk=1; #{half_ns} clk=0; #{half_ns} rst=0; #{half_ns};
 repeat({args.clocks}) begin clk=1; #{half_ns} clk=0; #{half_ns}; end
 $finish; end
endmodule
""")
    subprocess.run(["iverilog", "-g2012", "-o", work / "rtl.vvp", "-s", "cmp_tb", tb, *files], check=True, stdin=subprocess.DEVNULL)
    subprocess.run(["vvp", work / "rtl.vvp"], check=True, stdin=subprocess.DEVNULL, capture_output=True)
    # Sampling times: just before the next phase starts.
    times = [half_ns - 1 + half_ns * i for i in range(len(samples))]
    return parse_vcd(work / "rtl.vcd", times, regs)


def parse_vcd(path: Path, times: list[int], regs: dict[str, int] | None = None) -> dict[str, list[str]]:
    """Scoped name (without the dut) -> per sample time, its value string MSB first, extended to its width.
    regs: filled with the reg variables and their widths."""
    regs = {} if regs is None else regs
    ids: dict[str, list[tuple[str, int]]] = defaultdict(list)
    scope: list[str] = []
    current: dict[str, str] = {}
    result: dict[str, list[str]] = {}
    index = 0
    now = 0

    def snapshot() -> None:
        for code, names in ids.items():
            value = current.get(code, "x")
            for name, width in names:
                result.setdefault(name, []).append(value.rjust(width, "0" if value[0] in "01" else value[0]))

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("$scope"):
                scope.append(line.split()[2])
            elif line.startswith("$upscope"):
                scope.pop()
            elif line.startswith("$var"):
                fields = line.split()
                name = ".".join(scope[1:] + [fields[4]])  # scope[0] is the bench
                if name.startswith("dut."):
                    ids[fields[3]].append((name[4:], int(fields[2])))
                    if fields[1] == "reg":
                        regs[name[4:]] = int(fields[2])
            elif line.startswith("#"):
                t = int(line[1:])
                while index < len(times) and times[index] < t:
                    snapshot()
                    index += 1
                now = t
            elif line[0] in "01xz":
                current[line[1:]] = line[0]
            elif line[0] == "b":
                value, code = line[1:].split()
                current[code] = value
    while index < len(times):
        snapshot()
        index += 1
    return result


def rtl_check(args, work: Path, netlist: Netlist, labels: list[str], reference: dict[Bit, list[int]],
              measured: dict[Bit, list[int]], power_up: dict[Bit, int]) -> dict:
    """Named nets against the RTL, whose registers start where the flip-flops powered up, so X can't choose branches
    the gates don't. Registers the netlist lost start at 0, and memories stay X. reference and measured: bit -> per
    sample value."""
    regs: dict[str, int] = {}
    rtl_samples(args, work, regs=regs)
    deposits = {}
    for name, width in regs.items():
        bits = netlist.module["netnames"].get(name, {}).get("bits")
        value = ["0"] * width
        for i, bit in enumerate(bits or []):
            if i < width and bit in power_up:
                value[width - 1 - i] = str(power_up[bit])
        deposits[name] = "".join(value)
    rtl = rtl_samples(args, work, deposits)
    report = {"names": 0, "names_missing": 0, "bits": 0, "x_samples": 0, "netlist_vs_rtl": [], "redstone_vs_rtl": []}
    for name, entry in netlist.module["netnames"].items():
        if name.startswith("$") or name not in rtl:
            report["names_missing"] += not name.startswith("$")
            continue
        report["names"] += 1
        values = rtl[name]
        for i, bit in enumerate(entry["bits"]):
            if bit in ("0", "1", "x", "z"):
                continue
            report["bits"] += 1
            for label, target in (("netlist_vs_rtl", reference), ("redstone_vs_rtl", measured)):
                series = target.get(bit)
                if series is None:
                    continue
                wrong = []
                for s, v in enumerate(series):
                    if labels[s][0] == "h" or labels[s] in ("init", "rh"):
                        continue  # memories change in the low phase here, at the edge in RTL
                    ch = values[s][-1 - i] if i < len(values[s]) else "x"
                    if ch in "xz":
                        report["x_samples"] += 1
                    elif int(ch) != v:
                        wrong.append((s, int(ch), v))
                if wrong:
                    report[label].append({"name": f"{name}[{i}]", "bit": bit, "sample": labels[wrong[0][0]], "rtl": wrong[0][1], "got": wrong[0][2],
                                          "last": labels[wrong[-1][0]], "count": len(wrong)})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("--litematic", type=Path)
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--half", type=int, default=3000)
    parser.add_argument("--clocks", type=int, default=40)
    parser.add_argument("--mode", default="redstone")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reuse", action="store_true", help="read --out's headless.txt instead of running the schematic")
    parser.add_argument("--rtl", nargs="+", help="top module then its Verilog files")
    args = parser.parse_args()

    build = args.build
    litematic = args.litematic or build / "out.litematic"
    pins_file = args.pins or build / "pins"
    out_dir = args.out or build
    out_dir.mkdir(parents=True, exist_ok=True)

    terminals, nets = endpoints(build)
    netlist = Netlist(build / "netlist.json")
    region = load_region(litematic)
    live_pins = live_pins_on_wire(build, region, terminals, nets)
    for v in live_pins:
        print(f"[compare] FAULT: {sorted(v.nets)[0]} has no net and drives wire on {v.pos}: {region[v.pos].id}", flush=True)
    cell_names: dict[tuple, str] = {}
    cells: dict[str, tuple] = {}
    unreadable = Counter()
    on_nets = {key for source, sinks in nets.values() for key in (source, *sinks)}
    no_wire = []  # net ends with no wire at their cell at all
    for key, pos in terminals.items():
        block = region[pos].id if all(0 <= c < n for c, n in zip(pos, (region.width, region.height, region.length))) else "outside"
        if block not in READABLE:
            unreadable[block] += 1
            if key in on_nets:
                no_wire.append((key, pos, block))
            continue
        cell_names[key] = f"s{len(cell_names)}"
        cells[cell_names[key]] = pos
    json.dump({"names": [[*k, v] for k, v in cell_names.items()]}, open(out_dir / "cell_names.json", "w"))
    print(f"[compare] {len(cells)} cells of {len(terminals)} terminals, unreadable {dict(unreadable)}", flush=True)

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        work = Path(tmp)
        samples = run_headless(litematic, pins_file, cells, args.half, args.clocks, out_dir, args.mode, args.reuse)
        labels = [label for label, _ in samples]
        print(f"[compare] {len(samples)} samples", flush=True)
        report = analyse(args, work, netlist, nets, terminals, cell_names, samples, labels)
    net_of = {key: net_id for net_id, (source, sinks) in nets.items() for key in (source, *sinks)}
    report["no_wire"] = [dict(net=net_of[key], end=key, cell=pos, block=block) for key, pos, block in no_wire]
    report["summary"]["no_wire"] = len(no_wire)
    report["live_pins"] = [dict(pin=sorted(v.nets)[0], cell=v.pos) for v in live_pins]
    report["summary"]["live_pins"] = len(live_pins)
    json.dump(report, open(out_dir / "compare.json", "w"), indent=1, default=str)
    print(f"[compare] wrote {out_dir / 'compare.json'}")
    return 1 if report["summary"]["wiring"] or report["summary"]["gates"] or report["summary"]["flops"] or live_pins else 0


def analyse(args, work, netlist, nets, terminals, cell_names, samples, labels) -> dict:
    series = {key: [values.get(name) for _, values in samples] for key, name in cell_names.items()}
    n = len(samples)
    wiring = []
    for net_id, (source, sinks) in nets.items():
        src = series.get(source)
        if src is None:
            continue
        for sink in sinks:
            got = series.get(sink)
            if got is None:
                continue
            wrong = [s for s in range(n) if got[s] != src[s]]
            if wrong:
                wiring.append(dict(net=net_id, source=source, sink=sink, cell=terminals[sink], first=labels[wrong[0]], count=len(wrong),
                                   samples_differ=[labels[s] for s in wrong[:8]]))

    net_of = {}
    for net_id, (source, sinks) in nets.items():
        for key in (source, *sinks):
            net_of[key] = net_id
    layout = json.load(open(args.build / "component_layout.json"))
    component_list = json.load(open(args.build / "component_list.json"))
    # Pins tied to constants aren't on nets.
    constants = {(c["id"], p["name"]): p.get("const_value") for c in component_list["components"] for p in c["pins"]
                 if p.get("const_value") is not None}

    gates = []
    flops = []
    for comp in layout:
        pins = {p["name"]: p for p in comp["pins"]}
        kind, cid = comp["type"], comp["id"]

        def measure(pin: str) -> list | None:
            if (cid, pin) in constants:
                return [int(constants[cid, pin])] * n
            return series.get((cid, pin))

        if kind in ("AND", "OR", "XOR", "NOT", "MUX", "FULLADDER"):
            inputs = {p: measure(p) for p, d in pins.items() if d["direction"] == "IN"}
            if any(v is None for v in inputs.values()):
                continue
            for out in [p for p, d in pins.items() if d["direction"] == "OUT"]:
                got = measure(out)
                if got is None:
                    continue
                wrong = [s for s in range(n) if gate_output(kind, {p: v[s] for p, v in inputs.items()})[out] != got[s]]
                if wrong:
                    gates.append(dict(cell=cid, type=kind, pin=out, net=net_of.get((cid, out)), first=labels[wrong[0]], count=len(wrong)))
        elif kind in ("DFF", "DFFE"):
            d, q = measure("D"), measure("Q")
            e = measure("E") if kind == "DFFE" else [1] * n
            if d is None or q is None or e is None:
                continue
            bad = []
            for s in range(1, n):
                if labels[s][0] == "h" or labels[s] == "rh":  # took a rising edge since sample s - 1
                    want = d[s - 1] if e[s - 1] else q[s - 1]
                    if q[s] != want:
                        bad.append(s)
                elif q[s] != q[s - 1]:  # moved with no rising edge
                    bad.append(s)
            if bad:
                flops.append(dict(cell=cid, type=kind, net=net_of.get((cid, "Q")), first=labels[bad[0]], count=len(bad)))

    # The reference starts from the measured power-up of the flip-flops.
    for name, conn in netlist.flops.items():
        got = series.get((name, "Q"))
        netlist.values[conn["Q"][0]] = got[0] if got else 0
    netlist.values[netlist.module["ports"]["rst"]["bits"][0]] = 1
    netlist.settle_all()
    bits = bit_of_net(netlist, nets)
    expected: dict[Bit, list[int]] = defaultdict(list)

    def record() -> None:
        for bit in set(bits.values()):
            expected[bit].append(netlist.values[bit])

    record()
    netlist.rise(); record()
    netlist.fall(); record()
    netlist.set_input("rst", 0); record()
    for _ in range(args.clocks):
        netlist.rise(); record()
        netlist.fall(); record()

    versus = []
    sampled = {}
    for net_id, (source, sinks) in nets.items():
        bit = bits.get(net_id)
        if bit is None:
            continue
        src = series.get(source)
        if src is not None:
            sampled[bit] = src
        for key in (source, *sinks):
            got = series.get(key)
            if got is None:
                continue
            wrong = [s for s in range(n) if got[s] != expected[bit][s]]
            if wrong:
                versus.append(dict(net=net_id, end=key, role="source" if key == source else "sink", first=labels[wrong[0]],
                                   first_index=wrong[0], count=len(wrong)))
    versus.sort(key=lambda v: v["first_index"])
    report = dict(samples=labels, wiring=wiring, gates=gates, flops=flops, netlist=versus)
    if args.rtl:
        report["rtl"] = rtl_check(args, work, netlist, labels, expected, sampled, {b: v[0] for b, v in expected.items()})
    report["summary"] = dict(wiring=len(wiring), gates=len(gates), flops=len(flops), netlist_ends=len(versus),
                             netlist_first=versus[0]["first"] if versus else None,
                             rtl_vs_netlist=len(report["rtl"]["netlist_vs_rtl"]) if args.rtl else None,
                             redstone_vs_rtl=len(report["rtl"]["redstone_vs_rtl"]) if args.rtl else None)
    print("[compare]", report["summary"], flush=True)
    return report


if __name__ == "__main__":
    sys.exit(main())
