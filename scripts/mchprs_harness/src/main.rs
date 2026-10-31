// Usage: mchprs_harness <schem> <pins.txt> <script.txt> [modes]
// pins.txt: "<name> <x> <y> <z> <nx> <nz>" per line (pin repeater cell and its outward normal).
// script.txt: "settle N" | "set <name> 0|1" | "pulse <name> N" | "expect <name> 0|1" | "print <label>"
// modes: comma-separated noupdate, redstone, redpiler. All but noupdate update every block after
// pasting, as MCHPRS's /watch does.
use mchprs_blocks::blocks::{Block, LeverFace};
use mchprs_blocks::{BlockDirection, BlockPos};
use mchprs_redpiler::{BackendVariant, Compiler, CompilerOptions};
use mchprs_schematic::{load_schematic, paste_clipboard};
use mchprs_world::testing::TestWorld;
use mchprs_world::World;
use std::collections::BTreeMap;
use std::path::Path;

struct Runner {
    world: TestWorld,
    comp: Option<Compiler>,
}

impl Runner {
    fn tick(&mut self) {
        if let Some(c) = &mut self.comp {
            c.tick();
            c.flush(&mut self.world);
            return;
        }
        self.world.to_be_ticked.sort_by_key(|e| (e.ticks_left, e.tick_priority));
        for p in &mut self.world.to_be_ticked {
            p.ticks_left = p.ticks_left.saturating_sub(1);
        }
        while self.world.to_be_ticked.first().map_or(1, |e| e.ticks_left) == 0 {
            let e = self.world.to_be_ticked.remove(0);
            mchprs_redstone::tick(self.world.get_block(e.pos), &mut self.world, e.pos);
        }
    }

    fn toggle(&mut self, pos: BlockPos) {
        if let Some(c) = &mut self.comp {
            c.on_use_block(pos);
            c.flush(&mut self.world);
            return;
        }
        mchprs_redstone::on_use(self.world.get_block(pos), &mut self.world, pos);
    }
}

fn run(schem: &str, pins: &str, script: &str, mode: &str) -> usize {
    let cb = load_schematic(Path::new(schem)).unwrap();
    let chunks = |n: u32| (n as i32 + 24) / 16 + 1;
    let mut world = TestWorld::new(chunks(cb.size_x), chunks(cb.size_y), chunks(cb.size_z));
    let base = BlockPos::new(8, 8, 8);
    paste_clipboard(&mut world, &cb, base + BlockPos::new(cb.offset_x, cb.offset_y, cb.offset_z), false);
    let mut outs = BTreeMap::new();
    let mut levers = BTreeMap::new();
    for l in pins.lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<i32> = f[1..].iter().map(|v| v.parse().unwrap()).collect();
        let cell = base + BlockPos::new(n[0], n[1], n[2]);
        let lever = cell + BlockPos::new(n[3], 1, n[4]);
        if f[0] == "clk" {
            // builds leave the clock undriven; add the same stone + lever as other inputs
            world.set_block(cell + BlockPos::new(n[3], 0, n[4]), Block::Stone {});
            world.set_block(lever, Block::Lever { face: LeverFace::Floor, facing: BlockDirection::North, powered: false });
        }
        if matches!(world.get_block(lever), Block::Lever { .. }) {
            levers.insert(f[0].to_string(), lever);
        } else {
            outs.insert(f[0].to_string(), cell);
        }
    }
    if mode != "noupdate" {
        for x in 8..8 + cb.size_x as i32 {
            for y in 8..8 + cb.size_y as i32 {
                for z in 8..8 + cb.size_z as i32 {
                    let p = BlockPos::new(x, y, z);
                    mchprs_redstone::update(world.get_block(p), &mut world, p);
                }
            }
        }
    }
    let comp = (mode == "redpiler").then(|| {
        let mut c = Compiler::default();
        let opts = CompilerOptions { backend_variant: BackendVariant::Direct, ..Default::default() };
        let max = BlockPos::new(world.x_size * 16 - 1, world.y_size * 16 - 1, world.z_size * 16 - 1);
        c.compile(&world, (BlockPos::new(0, 0, 0), max), opts, world.to_be_ticked.clone(), Default::default());
        c
    });
    let mut r = Runner { world, comp };
    let mut fails = 0;
    let val = |r: &Runner, n: &str| -> u8 {
        match r.world.get_block(outs[n]) {
            Block::Repeater(repeater) => repeater.powered as u8,
            Block::RedstoneWire(wire) => (wire.power > 0) as u8,
            Block::RedstoneTorch { lit } | Block::RedstoneWallTorch { lit, .. } => lit as u8,
            b => panic!("{n} has no signal: {b:?}"),
        }
    };
    for line in script.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["settle", n] => (0..n.parse::<u32>().unwrap()).for_each(|_| r.tick()),
            ["set", n, v] => {
                let lever = levers[*n];
                let on = matches!(r.world.get_block(lever), Block::Lever { powered, .. } if powered);
                if on != (*v == "1") {
                    r.toggle(lever);
                }
            }
            ["pulse", n, w] => {
                let lever = levers[*n];
                r.toggle(lever);
                (0..w.parse::<u32>().unwrap()).for_each(|_| r.tick());
                r.toggle(lever);
            }
            ["expect", n, v] => {
                let got = val(&r, n);
                if got.to_string() != *v {
                    println!("  [{mode}] FAIL {n}: want {v} got {got}");
                    fails += 1;
                }
            }
            ["print", rest @ ..] => {
                let s: Vec<String> = outs.keys().map(|k| format!("{k}={}", val(&r, k))).collect();
                println!("  [{mode}] {:<22} {}", rest.join(" "), s.join(" "));
            }
            [] => {}
            _ => panic!("bad line {line}"),
        }
    }
    println!("[{mode}] {fails} failures");
    fails
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let pins = std::fs::read_to_string(&a[2]).unwrap();
    let script = std::fs::read_to_string(&a[3]).unwrap();
    let modes = a.get(4).map_or("redstone,redpiler", |m| m.as_str());
    let fails: usize = modes.split(',').map(|m| run(&a[1], &pins, &script, m)).sum();
    std::process::exit((fails > 0) as i32);
}
