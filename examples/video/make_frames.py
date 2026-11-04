"""Writes video.hex and pin_targets.json: a ball bouncing on a 16x12 screen, 32 frames, one 192-bit word per frame.

Bit 12 * (15 - column) + row of a word is the pixel at that column (0 at the left) and row (0 at the bottom), and is 1
where the pixel is dark, so a column's 12 bits are one output, dark0 to dark15: each pixel is a 3x3 tile of lamps (schematics/display_tile.litematic), which lights while its pin is low.
The ball crosses the screen and back once per loop, bouncing off the floor four times, and squashes where it lands.

The screen faces north (-z), 48x36 lamps with the tiles touching. Seen from the front, east is on the left.

Usage: python make_frames.py [--show]. --show prints the frames as text, # where lit.
"""
import json
import sys
from pathlib import Path

WIDTH, HEIGHT, FRAMES = 16, 12, 32
FLOOR = HEIGHT - 1  # the bottom row is the ground
BOUNCES = 4


BALL = [".##.", "####", "####", ".##."]
SQUASHED = [".####.", "######", "######"]  # where it lands
TOP = 6  # rows the ball rises between landings


def frame_bits(frame: int) -> list[list[int]]:
    phase = frame / FRAMES
    left = round((WIDTH - 4) * (1 - abs(1 - 2 * phase)))  # over and back
    t = frame * BOUNCES % FRAMES / FRAMES  # 0 at a landing, on towards the next
    bottom = FLOOR - 1 - round(4 * t * (1 - t) * TOP)
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


def dark_word(rows: list[list[int]]) -> int:
    """rows: top first."""
    return sum((1 - lit) << (HEIGHT * (WIDTH - 1 - c) + HEIGHT - 1 - r) for r, row in enumerate(rows) for c, lit in enumerate(row))


def pin_targets() -> dict[str, dict[str, int]]:
    """Each pixel's tile, lower left lamp at (x, y): column 0 furthest east, row 0 at the bottom. An output's pins stack
    up in bit order, as its column's tiles do."""
    return {f"dark{c}[{r}]": {"x": 3 * (WIDTH - 1 - c), "y": 3 * r + 1, "tile": 3}
            for r in range(HEIGHT) for c in range(WIDTH)}


def main() -> None:
    frames = [frame_bits(f) for f in range(FRAMES)]
    digits = WIDTH * HEIGHT // 4
    here = Path(__file__).parent
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
