#!/usr/bin/env python3
"""CNC (3-axis) versions of the Courmayeur model, written to results/cnc/.

A 3-axis router cuts from above only, so these are pure height fields: no tunnel bore
(an overhang), one piece 520 x 520 mm (or the same four 260 mm tiles, if your bed is small).

  heightmap_terrain_16bit.png     terrain + Dent du Géant, no roads/buildings   (recommended)
  heightmap_details_16bit.png     same + roads, buildings (they need a ≤0.5 mm tool to show)
  courmayeur_520x520_terrain.stl  solid with flat base, 0.4 mm grid (binary STL)
  results/cnc/tiles/courmayeur_<tile>_terrain.stl   same, cut into the four 260 x 260 mm tiles
  README.txt                      sizes and how to set them up in the CAM software

Run after build_tiles.py.
"""
import json, os, struct
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")    # generated models and viewer data
TILES_DIR = os.path.join(RESULTS, "tiles")
OUT = os.path.join(RESULTS, "cnc")
STL_STEP = 2            # every 2nd grid point -> 0.4 mm, plenty for a router


def write_stl(path, z, step_mm, x0=0.0, y0=0.0):
    """Binary STL of a closed solid: top surface z (row 0 = north), vertical sides, flat bottom at 0."""
    n, m = z.shape
    xs = x0 + np.arange(m) * step_mm
    ys = y0 + (n - 1 - np.arange(n)) * step_mm
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, z], -1)
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    tris = [np.stack([a, c, b], -2).reshape(-1, 3, 3), np.stack([b, c, d], -2).reshape(-1, 3, 3)]
    ring = np.concatenate([P[0, :], P[1:, -1], P[-1, -2::-1], P[-2:0:-1, 0]])
    r1 = np.roll(ring, -1, 0)
    rb, r1b = ring.copy(), r1.copy(); rb[:, 2] = 0; r1b[:, 2] = 0
    tris += [np.stack([ring, r1, r1b], 1), np.stack([ring, r1b, rb], 1)]
    ctr = np.array([xs.mean(), ys.mean(), 0.0])
    tris.append(np.stack([rb, r1b, np.broadcast_to(ctr, rb.shape)], 1))     # bottom, facing down
    T = np.concatenate(tris).astype(np.float32)
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(T), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, T
    with open(path, "wb") as f:
        f.write(b"Courmayeur terrain, millimetres".ljust(80, b" "))
        f.write(struct.pack("<I", len(T)))
        rec.tofile(f)
    return len(T)


def to_png16(z, path):
    lo, hi = float(z.min()), float(z.max())
    Image.fromarray(np.round((z - lo) / (hi - lo) * 65535).astype(np.uint16)).save(path)
    return lo, hi


def main():
    os.makedirs(os.path.join(OUT, "tiles"), exist_ok=True)
    meta = json.load(open(os.path.join(TILES_DIR, "viewer_meta.json")))
    n, step, size = meta["n"], meta["step_mm"], meta["size_mm"]
    elev = np.load(os.path.join(TILES_DIR, "elevation_m.npy")).astype(np.float64)
    terrain = (elev - meta["zmin_m"]) * meta["mm_per_m"] + meta["base_mm"]
    details = np.fromfile(os.path.join(TILES_DIR, "viewer_heights.bin"), "<u2").reshape(n, n) / 100.0

    lo, hi = to_png16(terrain, os.path.join(OUT, "heightmap_terrain_16bit.png"))
    lo2, hi2 = to_png16(details, os.path.join(OUT, "heightmap_details_16bit.png"))

    zs = terrain[::STL_STEP, ::STL_STEP]
    s = step * STL_STEP
    nt = write_stl(os.path.join(OUT, "courmayeur_520x520_terrain.stl"), zs, s)
    k = (zs.shape[0] - 1) // 2
    for name, (r, c) in {"NW": (0, 0), "NE": (0, 1), "SW": (1, 0), "SE": (1, 1)}.items():
        write_stl(os.path.join(OUT, "tiles", f"courmayeur_{name}_terrain.stl"),
                  zs[r * k:r * k + k + 1, c * k:c * k + k + 1], s)

    readme = f"""Courmayeur & Mont Blanc massif - CNC files
==========================================

Model size    {size:.0f} x {size:.0f} mm (X east, Y north), 20 x 20 km of ground, scale 1 : {1 / meta['mm_per_m'] * 1000:,.0f}
Total height  {hi:.1f} mm, of which {meta['base_mm']:.0f} mm is the solid base under the lowest valley
Relief        {hi - lo:.1f} mm from lowest valley (Dora Baltea, {meta['zmin_m']:.0f} m) to Mont Blanc (4 810 m)

Heightmaps (16-bit greyscale PNG, {n} x {n} px = {step} mm per pixel)
  heightmap_terrain_16bit.png   black = {lo:.2f} mm, white = {hi:.2f} mm above the bottom of the stock
  heightmap_details_16bit.png   black = {lo2:.2f} mm, white = {hi2:.2f} mm, with roads (+0.3 mm) and
                                buildings (3x real height); only worth it with a <=0.5 mm engraving tool
  Vectric Aspire / Carveco / VCarve:  import as bitmap/component, size {size:.0f} x {size:.0f} mm,
                                      Z height {hi - lo:.1f} mm, base height {lo:.1f} mm.
  Fusion 360 / other CAM:  use the STL instead.

STL (binary, mm, closed solid with flat bottom at Z = 0, {STL_STEP * step:.1f} mm grid, {nt:,} triangles)
  courmayeur_520x520_terrain.stl       the whole model in one piece
  tiles/courmayeur_<NW|NE|SW|SE>_terrain.stl   260 x 260 mm pieces, each with its own origin at its SW corner

Stock: at least {hi:.0f} mm thick (e.g. a {int(np.ceil(hi / 5) * 5)} mm board / laminated block), 520 x 520 mm + clamping margin.
Suggested toolpaths: roughing with a 6 mm flat end mill, finishing with a 3 mm then 1.5 mm (1/16") ball nose,
0.15-0.3 mm stepover. Very steep faces (Dent du Géant, Grandes Jorasses, north faces) are near-vertical:
use a long-reach tapered ball nose and expect the Dent du Géant tooth (1.6 x 0.9 mm) to need a 1 mm tool or be
added afterwards.

Not in the CNC files: the Mont Blanc Tunnel bore (an overhang a 3-axis machine can't cut). To show it, drill the
two portals horizontally by hand (2 mm drill), or mark them with paint. Portal positions in model mm:
"""
    tun = json.load(open(os.path.join(TILES_DIR, "viewer_tunnel.json")))
    for p in tun["portals"]:
        x, y, z = p["xyz"]
        readme += f"  {p['name']:<24} X {x:6.1f}  Y {y:6.1f}  Z {z:5.1f}\n"
    readme += "\nData: Regione Autonoma Valle d'Aosta (lidar DTM), IGN (RGE ALTI), INGV (TINITALY), © OpenStreetMap contributors.\n"
    open(os.path.join(OUT, "README.txt"), "w").write(readme)
    for f in sorted(os.listdir(OUT)) + [os.path.join("tiles", f) for f in sorted(os.listdir(os.path.join(OUT, "tiles")))]:
        p = os.path.join(OUT, f)
        if os.path.isfile(p):
            print(f"  results/cnc/{f}: {os.path.getsize(p) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
