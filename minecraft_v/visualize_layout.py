"""Render component layout to build/artifacts/layout.svg.

Usage:
    python -m minecraft_v.visualize_layout
    python -m minecraft_v.visualize_layout --layout build/artifacts/component_layout.json
"""
import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from minecraft_v.placement_ir import ComponentType

ARTIFACTS_DIR = Path("build/artifacts")

SCALE = 20  # pixels per block
PADDING = 40
LAYER_GAP = 60
LABEL_FONT = 7
PIN_RADIUS = 3

TYPE_COLOR = {
    ComponentType.INPUT_PIN:  "#66BB6A",
    ComponentType.OUTPUT_PIN: "#42A5F5",
    ComponentType.AND:        "#FFA726",
    ComponentType.OR:         "#EF5350",
    ComponentType.NOT:        "#AB47BC",
    ComponentType.XOR:        "#EC407A",
    ComponentType.DFF:        "#26C6DA",
    ComponentType.DFFE:       "#26A69A",
    ComponentType.DLATCH:     "#8D6E63",
    ComponentType.FULL_ADDER: "#78909C",
    ComponentType.MUX:        "#FFCA28",
    ComponentType.CUSTOM:     "#BDBDBD",
}


def _pin_world(origin: tuple, pin, footprint) -> tuple:
    ox, oy, oz = origin
    px, py, pz = pin.offset
    return (ox + px, oy + py, oz + (footprint.depth - 1 - pz))


def _svg_text(parent, x, y, text, font_size=LABEL_FONT, anchor="middle", fill="black", dy=0):
    el = ET.SubElement(parent, "text", x=str(x), y=str(y),
                       **{"font-size": str(font_size), "text-anchor": anchor,
                          "dominant-baseline": "central", "fill": fill,
                          "font-family": "monospace"})
    if dy:
        el.set("dy", str(dy))
    el.text = text
    return el


def _svg_rect(parent, x, y, w, h, fill, stroke="black", stroke_width=1, opacity=0.75, rx=2):
    ET.SubElement(parent, "rect", x=str(x), y=str(y), width=str(w), height=str(h),
                  fill=fill, stroke=stroke,
                  **{"stroke-width": str(stroke_width), "opacity": str(opacity), "rx": str(rx)})


def build_svg(layout: list[dict]) -> ET.Element:
    # Group placed entries by Y level; layout entries are self-contained
    by_y: defaultdict[int, list] = defaultdict(list)
    for entry in layout:
        by_y[entry["origin"][1]].append(entry)

    y_levels = sorted(by_y)

    def _fp(entry): return entry["footprint"]

    # Compute per-layer canvas size (X-Z plane)
    layer_dims: list[tuple[int, int]] = []
    for y in y_levels:
        max_x = max(e["origin"][0] + _fp(e)["width"]  for e in by_y[y]) + 2
        max_z = max(e["origin"][2] + _fp(e)["depth"]  for e in by_y[y]) + 2
        layer_dims.append((max_x, max_z))

    canvas_w = max(w for w, _ in layer_dims) * SCALE + PADDING * 2
    canvas_h = sum(h * SCALE + PADDING * 2 + LAYER_GAP for _, h in layer_dims) + 40

    svg = ET.Element("svg",
                     xmlns="http://www.w3.org/2000/svg",
                     width=str(int(canvas_w)),
                     height=str(int(canvas_h)))

    ET.SubElement(svg, "rect", x="0", y="0",
                  width=str(int(canvas_w)), height=str(int(canvas_h)),
                  fill="white")

    y_offset = PADDING

    for layer_idx, y in enumerate(y_levels):
        _, layer_h = layer_dims[layer_idx]
        layer_pixel_h = layer_h * SCALE

        _svg_text(svg, canvas_w / 2, y_offset - 14,
                  f"Y = {y}", font_size=11, fill="#444")

        for gx in range(0, layer_dims[layer_idx][0] + 1):
            for gz in range(0, layer_h + 1):
                ET.SubElement(svg, "circle",
                               cx=str(int(PADDING + gx * SCALE)),
                               cy=str(int(y_offset + gz * SCALE)),
                               r="1", fill="#ddd")

        for entry in by_y[y]:
            cid = entry["id"]
            ox, _, oz = entry["origin"]
            fp = entry["footprint"]
            fw, fd = fp["width"], fp["depth"]
            comp_type = ComponentType(entry["type"])
            color = TYPE_COLOR.get(comp_type, "#BDBDBD")

            sx = PADDING + ox * SCALE
            sz = y_offset + oz * SCALE
            sw = fw * SCALE
            sh = fd * SCALE

            _svg_rect(svg, sx, sz, sw, sh, fill=color)

            _svg_text(svg, sx + sw / 2, sz + sh / 2 - LABEL_FONT,
                      comp_type.value, font_size=LABEL_FONT, fill="#111")
            short_id = cid if len(cid) <= 14 else cid[-14:]
            _svg_text(svg, sx + sw / 2, sz + sh / 2 + LABEL_FONT,
                      short_id, font_size=max(5, LABEL_FONT - 1), fill="#333")

            for pin in entry["pins"]:
                px, _, pz = pin["offset"]
                # mirror Z to match placement_engine._pin_world
                pz_w = oz + (fd - 1 - pz)
                px_w = ox + px
                pcx = PADDING + (px_w + 0.5) * SCALE
                pcy = y_offset + (pz_w + 0.5) * SCALE
                const_val = pin.get("const_value")
                if const_val == "1":
                    pin_color = "#e65100"  # orange = driven high
                elif const_val == "0":
                    pin_color = "#78909c"  # grey = tied low
                elif pin["direction"] == "OUT":
                    pin_color = "#c62828"
                else:
                    pin_color = "#1565c0"
                ET.SubElement(svg, "circle",
                               cx=str(int(pcx)), cy=str(int(pcy)),
                               r=str(PIN_RADIUS), fill=pin_color,
                               stroke="white", **{"stroke-width": "1"})
                cx_comp = PADDING + (ox + fw / 2) * SCALE
                cz_comp = y_offset + (oz + fd / 2) * SCALE
                dx_label = 1 if pcx >= cx_comp else -1
                dz_label = 1 if pcy >= cz_comp else -1
                label = pin["name"] if const_val is None else f"{pin['name']}={const_val}"
                _svg_text(svg,
                          pcx + dx_label * (PIN_RADIUS + 2),
                          pcy + dz_label * (PIN_RADIUS + 2),
                          label,
                          font_size=5,
                          anchor="start" if dx_label > 0 else "end",
                          fill=pin_color)

        y_offset += layer_pixel_h + PADDING + LAYER_GAP

    # Legend
    legend_x = PADDING
    legend_y = y_offset
    _svg_text(svg, legend_x, legend_y, "Legend:", font_size=9, anchor="start", fill="#333")
    legend_y += 14
    for t, color in TYPE_COLOR.items():
        _svg_rect(svg, legend_x, legend_y - 8, 12, 10, fill=color, opacity=0.9)
        _svg_text(svg, legend_x + 16, legend_y - 2, t.value, font_size=8, anchor="start")
        legend_x += 100
        if legend_x + 100 > canvas_w:
            legend_x = PADDING
            legend_y += 16

    return svg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default=str(ARTIFACTS_DIR / "component_layout.json"))
    parser.add_argument("--out", default=str(ARTIFACTS_DIR / "layout.svg"))
    args = parser.parse_args()

    layout = json.loads(Path(args.layout).read_text())
    svg = build_svg(layout)

    ET.indent(svg, space="  ")
    tree = ET.ElementTree(svg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path), encoding="unicode", xml_declaration=False)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
