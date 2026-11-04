"""Writes video.hex and pin_targets.json for a 16x12 screen of 128 frames, one 192-bit word per frame.

The frames come from badapple.hex, or another file in its layout given with --from: a hex number per line, a frame each,
whose bit 16 * y + x is 1 where the pixel x from the left and y from the top is dark. Fewer than 128 frames hold the
last one. --ball draws a bouncing ball instead: it crosses the screen and back twice, bouncing eight times a crossing,
lower each time, squashing where it lands.

video.hex lays a frame out for the memory: bit 12 * (15 - x) + 11 - y is that pixel, 1 where dark, so a column's 12
bits are one output, dark0 to dark15, in the order the memory's bits run. Each pixel is a 3x3 tile of lamps
(schematics/display_tile.litematic), which lights while its pin is low.

The screen faces north (-z), 48x36 lamps with the tiles touching. Seen from the front, east is on the left.

Usage: python make_frames.py [--from FILE | --ball] [--show]. --show prints the frames as text, # where lit.
"""
import json
import sys
from pathlib import Path

WIDTH, HEIGHT, FRAMES = 16, 12, 128
FLOOR = HEIGHT - 1  # the bottom row is the ground
CROSSINGS = 4  # over, back, over, back
BOUNCE_FRAMES = 8
TOPS = [7, 6, 5, 4, 3, 3, 2, 2]  # rows the ball rises, bounce by bounce across the screen

BALL = [".##.", "####", "####", ".##."]
SQUASHED = [".####.", "######", "######"]  # where it lands


def frame_bits(frame: int) -> list[list[int]]:
    per_crossing = FRAMES // CROSSINGS
    crossing, along = divmod(frame, per_crossing)
    step = along / per_crossing
    left = round((WIDTH - 4) * (step if crossing % 2 == 0 else 1 - step))
    bounce, since = divmod(along, BOUNCE_FRAMES)
    t = since / BOUNCE_FRAMES  # 0 at a landing, on towards the next
    bottom = FLOOR - 1 - round(4 * t * (1 - t) * TOPS[bounce])
    sprite = BALL if t else SQUASHED
    if not t:
        left -= 1
    rows = [[0] * WIDTH for _ in range(HEIGHT)]
    rows[FLOOR] = [1] * WIDTH
    for r, line in enumerate(sprite):
        for c, ch in enumerate(line):
            if ch == "#" and 0 <= left + c < WIDTH:
                rows[bottom - len(sprite) + 1 + r][left + c] = 1
    return rows


def from_file(path: Path) -> list[list[list[int]]]:
    words = [int(line, 16) for line in path.read_text().split()]
    if not 0 < len(words) <= FRAMES:
        raise SystemExit(f"{path}: {len(words)} frames, not 1 to {FRAMES}")
    if any(w >> WIDTH * HEIGHT for w in words):
        raise SystemExit(f"{path}: a frame wider than {WIDTH * HEIGHT} bits")
    words += [words[-1]] * (FRAMES - len(words))
    return [[[1 - (w >> (WIDTH * y + x) & 1) for x in range(WIDTH)] for y in range(HEIGHT)] for w in words]


def dark_word(rows: list[list[int]]) -> int:
    """rows: top first."""
    return sum((1 - lit) << (HEIGHT * (WIDTH - 1 - c) + HEIGHT - 1 - r)
               for r, row in enumerate(rows) for c, lit in enumerate(row))


def pin_targets() -> dict[str, dict[str, int]]:
    """Each pixel's tile, lower left lamp at (x, y): column 0 furthest east, row 0 at the bottom. An output's pins stack
    up in bit order, as its column's tiles do."""
    return {f"dark{c}[{r}]": {"x": 3 * (WIDTH - 1 - c), "y": 3 * r + 1, "tile": 3}
            for r in range(HEIGHT) for c in range(WIDTH)}


def main() -> None:
    here = Path(__file__).parent
    if "--ball" in sys.argv:
        frames = [frame_bits(f) for f in range(FRAMES)]
    else:
        source = Path(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else here / "badapple.hex"
        frames = from_file(source)
    digits = WIDTH * HEIGHT // 4
    (here / "video.hex").write_text("".join(f"{dark_word(f):0{digits}x}\n" for f in frames))
    targets = pin_targets()
    (here / "pin_targets.json").write_text("{\n" + ",\n".join(f"  {json.dumps(pin)}: {json.dumps(at)}" for pin, at in targets.items())
                                           + "\n}\n")
    if "--show" in sys.argv:
        for n, rows in enumerate(frames):
            print(f"frame {n}")
            print("\n".join("".join("#" if b else "." for b in row) for row in rows))


if __name__ == "__main__":
    main()
