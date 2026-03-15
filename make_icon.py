#!/usr/bin/env python3
"""
Generate AutoNote app icons for all platforms.
Requires: pip install pillow

Output:
  assets/icon.png   — 512×512 master
  assets/icon.ico   — multi-size Windows icon
  assets/icon.icns  — macOS icon (uses iconutil on macOS, PNG fallback elsewhere)
"""
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).parent / "assets"


def _draw(size: int):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)
    s   = size

    # ── Background: dark-teal rounded square ──────────────────────────────────
    mg = max(int(s * 0.04), 2)
    d.rounded_rectangle(
        [mg, mg, s - mg, s - mg],
        radius=int(s * 0.20),
        fill=(13, 38, 38, 255),          # #0D2626
    )

    # ── White note / paper shape ──────────────────────────────────────────────
    pw = int(s * 0.52)
    ph = int(s * 0.58)
    px = (s - pw) // 2 - int(s * 0.02)
    py = (s - ph) // 2 + int(s * 0.02)
    d.rounded_rectangle(
        [px, py, px + pw, py + ph],
        radius=max(int(s * 0.055), 2),
        fill=(228, 244, 244, 255),        # near-white, slight teal tint
    )

    # ── Cyan ruled lines (3 full + 1 short) ───────────────────────────────────
    lh   = max(int(s * 0.030), 2)
    gap  = int(ph * 0.152)
    ly0  = py + int(ph * 0.20)
    lx0  = px + int(pw * 0.15)
    lx1f = px + int(pw * 0.83)   # full-length end
    lx1s = px + int(pw * 0.53)   # short-line end
    for i in range(4):
        ly  = ly0 + i * gap
        lx1 = lx1f if i < 3 else lx1s
        d.rounded_rectangle(
            [lx0, ly, lx1, ly + lh],
            radius=lh // 2,
            fill=(77, 208, 225, 200),     # #4DD0E1
        )

    # ── 4-point star sparkle (top-right of paper) ─────────────────────────────
    scx = px + pw + int(s * 0.055)
    scy = py  - int(s * 0.038)
    so  = int(s * 0.082)
    si  = int(s * 0.032)
    pts = []
    for i in range(8):
        angle = math.pi * i / 4 - math.pi / 2
        r     = so if i % 2 == 0 else si
        pts.append((scx + r * math.cos(angle), scy + r * math.sin(angle)))
    d.polygon(pts, fill=(0, 229, 255, 255))   # #00E5FF

    return img


def main():
    OUT.mkdir(exist_ok=True)

    # ── 512-px master PNG ─────────────────────────────────────────────────────
    base = _draw(512)
    base.save(OUT / "icon.png")
    print("  assets/icon.png  (512×512)")

    # ── Windows .ico (multiple sizes in one file) ─────────────────────────────
    from PIL import Image
    ico_sizes = [256, 128, 64, 48, 32, 16]
    frames    = [_draw(s).resize((s, s), Image.LANCZOS) for s in ico_sizes]
    frames[0].save(
        OUT / "icon.ico", format="ICO",
        append_images=frames[1:],
        sizes=[(s, s) for s in ico_sizes],
    )
    print("  assets/icon.ico  (multi-size)")

    # ── macOS .icns ───────────────────────────────────────────────────────────
    if sys.platform == "darwin":
        tmp = Path(tempfile.mkdtemp())
        iconset = tmp / "AutoNote.iconset"
        iconset.mkdir()
        for sz, scale in [(16,1),(16,2),(32,1),(32,2),(128,1),(128,2),
                          (256,1),(256,2),(512,1),(512,2)]:
            actual = sz * scale
            suffix = "@2x" if scale == 2 else ""
            _draw(actual).save(iconset / f"icon_{sz}x{sz}{suffix}.png")
        result = subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(OUT / "icon.icns")],
            capture_output=True,
        )
        shutil.rmtree(tmp)
        if result.returncode == 0:
            print("  assets/icon.icns")
        else:
            print(f"  [warn] iconutil failed: {result.stderr.decode()}")
    else:
        # CI non-macOS: copy PNG — macOS runner will regenerate the real .icns
        shutil.copy(OUT / "icon.png", OUT / "icon.icns")
        print("  assets/icon.icns (PNG placeholder, macOS runner will replace)")


if __name__ == "__main__":
    main()
