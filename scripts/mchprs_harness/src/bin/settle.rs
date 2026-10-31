// Usage: settle <schem> <cuts.txt> <states.txt>
// Removes the blocks listed in cuts.txt ("x y z" per line), updates every block and runs the
// redstone until nothing is pending, then writes each signal block's state:
// "x y z powered <0|1> <0|1>" (repeater: powered, locked) | "x y z power <n>" (dust) | "x y z lit <0|1>"
use mchprs_blocks::blocks::Block;
use mchprs_blocks::BlockPos;
use mchprs_schematic::{load_schematic, paste_clipboard};
use mchprs_world::testing::TestWorld;
use mchprs_world::World;
use std::fmt::Write;
use std::path::Path;

const MAX_TICKS: u32 = 20000;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cb = load_schematic(Path::new(&a[1])).unwrap();
    let chunks = |n: u32| (n as i32 + 24) / 16 + 1;
    let mut world = TestWorld::new(chunks(cb.size_x), chunks(cb.size_y), chunks(cb.size_z));
    let base = BlockPos::new(8, 8, 8);
    paste_clipboard(&mut world, &cb, base + BlockPos::new(cb.offset_x, cb.offset_y, cb.offset_z), false);
    for l in std::fs::read_to_string(&a[2]).unwrap().lines() {
        let n: Vec<i32> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
        world.set_block(base + BlockPos::new(n[0], n[1], n[2]), Block::Air {});
    }
    let (sx, sy, sz) = (cb.size_x as i32, cb.size_y as i32, cb.size_z as i32);
    for x in 0..sx {
        for y in 0..sy {
            for z in 0..sz {
                let p = base + BlockPos::new(x, y, z);
                mchprs_redstone::update(world.get_block(p), &mut world, p);
            }
        }
    }
    let mut ticks = 0;
    while !world.to_be_ticked.is_empty() && ticks < MAX_TICKS {
        world.to_be_ticked.sort_by_key(|e| (e.ticks_left, e.tick_priority));
        for p in &mut world.to_be_ticked {
            p.ticks_left = p.ticks_left.saturating_sub(1);
        }
        while world.to_be_ticked.first().map_or(1, |e| e.ticks_left) == 0 {
            let e = world.to_be_ticked.remove(0);
            mchprs_redstone::tick(world.get_block(e.pos), &mut world, e.pos);
        }
        ticks += 1;
    }
    println!("settled in {ticks} ticks, {} still pending", world.to_be_ticked.len());
    let mut out = String::new();
    for x in 0..sx {
        for y in 0..sy {
            for z in 0..sz {
                let _ = match world.get_block(base + BlockPos::new(x, y, z)) {
                    Block::Repeater(r) => writeln!(out, "{x} {y} {z} powered {} {}", r.powered as u8, r.locked as u8),
                    Block::RedstoneWire(w) => writeln!(out, "{x} {y} {z} power {}", w.power),
                    Block::RedstoneTorch { lit } | Block::RedstoneWallTorch { lit, .. } | Block::RedstoneLamp { lit } => {
                        writeln!(out, "{x} {y} {z} lit {}", lit as u8)
                    }
                    _ => Ok(()),
                };
            }
        }
    }
    std::fs::write(&a[3], out).unwrap();
}
