"""Pin-face repro: n short RAM blocks sharing address, write and data pins, one gate. 12 blocks, 11 address bits fails in minutes.

Usage: ram_pin_repro.py <out> <n> <abits> <seed> [limit_s]. Env: GROUP, ROW, W, RAM_ADDR_PITCH (even only), RAM_CHANNEL.
"""
import json, os, re, subprocess, sys, time
out, n, ab, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else 120
ROW, W = int(os.environ.get("ROW", 2)), int(os.environ.get("W", 1))
os.makedirs(out, exist_ok=True)
next_bit = 2; ports = {}; bits = {}
def port(name, width, d):
    global next_bit
    b = list(range(next_bit, next_bit + width)); next_bit += width
    ports[name] = {"direction": d, "bits": b}; return b
sh = {p: port(p, ab, "input") for p in ("aa", "ab", "ac", "ad", "wa")}
sh["we"] = port("we", 1, "input"); sh["wd"] = port("wd", W, "input")
cells = {}
if os.environ.get("GROUP"):  # one block with lanes, in place of n blocks
    conn = dict(sh)
    for r in ("ra", "rb", "rc", "rd"):
        conn[r] = port(r, n * W, "output")
    cells["group"] = {"hide_name": 0, "type": "ram_group", "parameters": {
        "ROW_BITS": format(ROW, "032b"), "BANKS": format(1, "032b"), "STACKS": format(n, "032b"), "ABITS": format(ab, "032b"),
        "WIDTH": format(W, "032b")}, "connections": conn}
for k in range(0 if os.environ.get("GROUP") else n):
    conn = dict(sh)
    for r in ("ra", "rb", "rc", "rd"):
        conn[r] = port(f"m{k}{r}", W, "output")
    cells[f"m{k}"] = {"hide_name": 0, "type": "ram_stack", "parameters": {
        "ROW_BITS": format(ROW, "032b"), "BANKS": format(1, "032b"), "BASE": format(k, "032b"), "ABITS": format(ab, "032b"),
        "WIDTH": format(W, "032b"), "INIT": "0" * ((1 << ROW) * W)}, "connections": conn}
y = port("nwe", 1, "output")
cells["keep_placer_happy"] = {"hide_name": 0, "type": "$_NOT_", "parameters": {}, "connections": {"A": sh["we"], "Y": y}}
json.dump({"creator": "mini", "modules": {"main": {"ports": ports, "cells": cells, "netnames": {}}}}, open(out + "/netlist.json", "w"))
t = time.time(); env = {**os.environ, "MINECRAFT_V_ARTIFACTS_DIR": out}
try:
    r = subprocess.run([".venv/bin/python", "scripts/build_with_pins.py", out + "/pins", "--netlist", out + "/netlist.json", "--route-passes", "1",
                        "--generate-clock", "none", "--place-seed", seed, "--out-litematic", out + "/out.litematic"], capture_output=True, text=True, env=env, timeout=limit)
    rc, log = r.returncode, r.stdout + r.stderr
except subprocess.TimeoutExpired as e:
    rc, log = "timeout", ((e.stdout or b"").decode() + (e.stderr or b"").decode())
open(out + "/log", "w").write(log)
wk = re.findall(r"workspace (\S+)", log); err = re.findall(r"^RuntimeError: (.{0,70})|^ValueError: (.{0,70})", log, re.M)
print(f"group={os.environ.get('GROUP', 0)} channel={os.environ.get('RAM_CHANNEL',24)} row={ROW} w={W} pitch={os.environ.get('RAM_ADDR_PITCH', 4)} n={n} ab={ab} seed={seed} rc={rc} {time.time()-t:.0f}s {wk[-1] if wk else '?'} searches_failed={len(re.findall(r'^.wire. net_', log, re.M))} {''.join(err[-1]) if err else ''}", flush=True)
