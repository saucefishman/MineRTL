# Test designs

Small circuits kept for acceptance scripts in `scripts/dev/`, not as examples. Build one with `./synth.sh tests/designs/<name>`.

| Design | Checks |
| --- | --- |
| `powerup` | Flip-flops keep their initial values at power-up only when the build is settled (`--no-settle` loses them on `--place-seed 7`). `scripts/dev/check_powerup.py`. |
| `powerup_inverted` | Clocks that rest high at power-up: settling has to keep every initial value. `scripts/dev/check_powerup.py`. |
| `preload` | Two preloaded tiled memories on shared addresses, read back before any write. `scripts/dev/check_preload.py`. |
