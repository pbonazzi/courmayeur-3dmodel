#!/usr/bin/env python3
"""Full-colour versions of the four final tiles (for full-colour printing, e.g. on Craftcloud).

Colours follow the real landscape, from OpenStreetMap (data/osm/landcover.json): glaciers white,
forests dark green, bare rock / scree grey, lakes and rivers blue; elsewhere alpine meadow
shading with elevation, rock on very steep ground. Roads, buildings, the highlighted house,
tunnel bore, walls and the engraved texts get their own colours.

Upload files are simplified (manifold3d, max deviation SIMPLIFY_MM) to ~1 M triangles per tile:
online print services choke on 3.5 M-triangle meshes. Writes
  results/final_color/courmayeur_<tile>_26x26cm_color.3mf   per-vertex colours (3MF materials extension)
  results/final_color/courmayeur_<tile>_26x26cm_color.wrl   VRML97 with per-vertex colours
  results/final_stl/courmayeur_<tile>_26x26cm.stl           single-colour, binary STL
  results/final/courmayeur_<tile>_26x26cm.obj               single-colour OBJ
  results/tiles/viewer_colors.bin, results/tiles/viewer_mesh_<tile>_col.bin   the same colours for the viewer
Run after build_tiles.py.
"""
import io, json, os, sys, zipfile
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")          # downloaded inputs (fetch_data.py)
RESULTS = os.path.join(HERE, "results")    # generated models and viewer data
sys.path.insert(0, HERE)
import build_tiles as B

TILES_DIR = os.path.join(RESULTS, "tiles")
SIMPLIFY_MM = 0.05
OUT = os.path.join(RESULTS, "final_color")

GLACIER, FOREST, ROCK, WATER = 1, 2, 3, 4
PALETTE = {
    "meadow_low": (109, 140, 74), "meadow_mid": (140, 157, 92), "meadow_high": (163, 162, 117),
    "rock": (141, 132, 122), "rock_steep": (120, 112, 104), "forest": (61, 90, 46),
    "glacier": (238, 242, 246), "water": (91, 141, 184),
    "road": (85, 85, 85), "building": (176, 71, 47), "highlight": (229, 176, 11),
    "wall": (59, 55, 51), "bore": (26, 26, 26), "text": (241, 230, 200),
}


# ---------------------------------------------------------------- landcover raster
def _rings(members):
    """Join way segments (lists of (x, y)) into closed rings."""
    segs = [list(m) for m in members if len(m) > 1]
    rings = []
    while segs:
        ring = segs.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, s in enumerate(segs):
                if s[0] == ring[-1]:
                    ring += s[1:]
                elif s[-1] == ring[-1]:
                    ring += s[-2::-1]
                elif s[-1] == ring[0]:
                    ring = s[:-1] + ring
                elif s[0] == ring[0]:
                    ring = s[:0:-1] + ring
                else:
                    continue
                segs.pop(i); changed = True
                break
        if len(ring) > 3:
            rings.append(ring)
    return rings


def landcover_grid():
    n = B.N
    t = Transformer.from_crs(4326, 32632, always_xy=True)
    x0, y0, st = B.CX - B.SIZE_M / 2, B.CY + B.SIZE_M / 2, B.GSTEP

    def px(geom):
        lon = np.array([p["lon"] for p in geom]); lat = np.array([p["lat"] for p in geom])
        x, y = t.transform(lon, lat)
        return [(round(a, 2), round(b, 2)) for a, b in zip((x - x0) / st, (y0 - y) / st)]

    def kind(tags):
        v = tags.get("natural") or tags.get("landuse") or tags.get("waterway")
        return {"glacier": GLACIER, "wood": FOREST, "forest": FOREST, "bare_rock": ROCK, "scree": ROCK,
                "rock": ROCK, "shingle": ROCK, "water": WATER, "riverbank": WATER}.get(v)

    layers = {k: Image.new("L", (n, n), 0) for k in (FOREST, ROCK, WATER, GLACIER)}
    for e in json.load(open(os.path.join(DATA, "osm", "landcover.json")))["elements"]:
        k = kind(e["tags"])
        if not k:
            continue
        img = Image.new("L", (n, n), 0); d = ImageDraw.Draw(img)
        if e["type"] == "way" and len(e.get("geometry", [])) > 3:
            d.polygon(px(e["geometry"]), fill=1)
        elif e["type"] == "relation":
            outer = [px(m["geometry"]) for m in e.get("members", []) if m.get("role") != "inner" and m.get("geometry")]
            inner = [px(m["geometry"]) for m in e.get("members", []) if m.get("role") == "inner" and m.get("geometry")]
            for r in _rings(outer):
                d.polygon(r, fill=1)
            for r in _rings(inner):
                d.polygon(r, fill=0)
        else:
            continue
        layers[k] = Image.fromarray(np.maximum(np.asarray(layers[k]), np.asarray(img)))
    lc = np.zeros((n, n), np.uint8)
    for k in (FOREST, ROCK, WATER, GLACIER):            # later wins: glaciers on top
        lc[np.asarray(layers[k]) > 0] = k
    return lc


# ---------------------------------------------------------------- colours on the grid
def grid_colors(elev, feat, lc):
    P = {k: np.array(v, float) for k, v in PALETTE.items()}
    gy, gx = np.gradient(elev, B.GSTEP)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # meadow by elevation, fading to rock above ~2500 m
    e = elev[..., None]
    col = np.where(e < 1800, P["meadow_low"] + (P["meadow_mid"] - P["meadow_low"]) * np.clip((e - 900) / 900, 0, 1),
                   P["meadow_mid"] + (P["meadow_high"] - P["meadow_mid"]) * np.clip((e - 1800) / 600, 0, 1))
    col = col + (P["rock"] - col) * np.clip((e - 2450) / 300, 0, 1)
    col = np.where((slope > 42)[..., None], P["rock_steep"], col)
    for k, name in ((FOREST, "forest"), (ROCK, "rock"), (WATER, "water"), (GLACIER, "glacier")):
        col[lc == k] = P[name]
    col[feat == 1] = P["road"]; col[feat == 2] = P["building"]; col[feat == 3] = P["highlight"]
    return np.clip(np.round(col), 0, 255).astype(np.uint8)


def vertex_colors(v, name, Z, G):
    """Colour of each vertex of a final tile mesh (same rules as the viewer)."""
    tile = B.MODEL_MM / B.TILES
    tr, tc = (0 if name[0] == "N" else 1), (0 if name[1] == "W" else 1)
    n, st = Z.shape[0], B.STEP_MM
    lx, ly, z = v[:, 0], v[:, 1], v[:, 2]
    X, Y = lx + tc * tile, ly + (B.TILES - 1 - tr) * tile
    cf, rf = X / st, (n - 1) - Y / st
    c, r = np.clip(np.round(cf).astype(int), 0, n - 1), np.clip(np.round(rf).astype(int), 0, n - 1)
    on_grid = (np.abs(cf - np.round(cf)) < 1e-3) & (np.abs(rf - np.round(rf)) < 1e-3)
    surf = Z[r, c]
    in_wall = np.minimum.reduce([lx, ly, tile - lx, tile - ly])
    out = G[r, c].copy()
    text = ~on_grid & (in_wall > 1e-3) & (in_wall < 1.0) & (z < 40)
    side = (z < 1e-4) | ((in_wall < 1e-3) & (~on_grid | (z < surf - 1e-3)))
    bore = ~on_grid | (z < surf - 0.01)
    out[bore] = PALETTE["bore"]
    out[side] = PALETTE["wall"]
    out[text] = PALETTE["text"]
    return out


# ---------------------------------------------------------------- writers
def write_3mf(path, v, f, col):
    hexes = np.array([f"#{a:02X}{b:02X}{c:02X}" for a, b, c in map(tuple, np.unique(col, axis=0))])
    uniq, idx = np.unique(col, axis=0, return_inverse=True)
    idx = idx.ravel()
    buf = io.StringIO()
    buf.write('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
              'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
              'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">\n'
              '<metadata name="Title">Courmayeur terrain tile</metadata>\n<resources>\n<m:colorgroup id="1">\n')
    buf.write("".join(f'<m:color color="{h}"/>\n' for h in hexes))
    buf.write('</m:colorgroup>\n<object id="2" type="model" pid="1" pindex="0">\n<mesh>\n<vertices>\n')
    np.savetxt(buf, v, fmt='<vertex x="%.3f" y="%.3f" z="%.3f"/>')
    buf.write('</vertices>\n<triangles>\n')
    tri = np.column_stack([f, idx[f[:, 0]], idx[f[:, 1]], idx[f[:, 2]]])
    np.savetxt(buf, tri, fmt='<triangle v1="%d" v2="%d" v3="%d" pid="1" p1="%d" p2="%d" p3="%d"/>')
    buf.write('</triangles>\n</mesh>\n</object>\n</resources>\n<build>\n<item objectid="2"/>\n</build>\n</model>\n')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
                   'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", buf.getvalue())


def write_wrl(path, v, f, col):
    with open(path, "w") as fh:
        fh.write("#VRML V2.0 utf8\n# Courmayeur terrain tile, units: millimetres\n"
                 "Shape {\n appearance Appearance { material Material { diffuseColor 1 1 1 } }\n"
                 " geometry IndexedFaceSet {\n  solid TRUE\n  colorPerVertex TRUE\n  coord Coordinate { point [\n")
        np.savetxt(fh, v, fmt="%.3f %.3f %.3f,")
        fh.write("  ] }\n  color Color { color [\n")
        np.savetxt(fh, col / 255.0, fmt="%.3f %.3f %.3f,")
        fh.write("  ] }\n  coordIndex [\n")
        np.savetxt(fh, f, fmt="%d,%d,%d,-1,")
        fh.write("  ]\n }\n}\n")


def write_stl(path, v, f):
    import struct
    T = v[f].astype(np.float32)
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(T), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, T
    with open(path, "wb") as fh:
        fh.write(b"Courmayeur terrain tile, millimetres".ljust(80, b" "))
        fh.write(struct.pack("<I", len(T)))
        rec.tofile(fh)


def write_obj(path, v, f):
    with open(path, "w") as fh:
        fh.write(f"# Courmayeur terrain tile, millimetres, +Y = north\no {os.path.splitext(os.path.basename(path))[0]}\n")
        np.savetxt(fh, v, fmt="v %.3f %.3f %.3f")
        np.savetxt(fh, f + 1, fmt="f %d %d %d")


def simplify(v, f):
    import manifold3d as mf
    m = mf.Manifold(mf.Mesh(vert_properties=v.astype(np.float32), tri_verts=f.astype(np.uint32))).simplify(SIMPLIFY_MM)
    assert m.status() == mf.Error.NoError, m.status()
    out = m.to_mesh()
    return np.asarray(out.vert_properties)[:, :3].astype(np.float64), np.asarray(out.tri_verts).astype(np.int64)


def load_mesh(name):
    d = open(os.path.join(TILES_DIR, f"viewer_mesh_{name}.bin"), "rb").read()
    nv, nf = np.frombuffer(d[:8], "<i4")
    v = np.frombuffer(d, "<f4", nv * 3, 8).reshape(-1, 3).astype(np.float64)
    f = np.frombuffer(d, "<u4", nf * 3, 8 + nv * 12).reshape(-1, 3).astype(np.int64)
    return v, f


def main():
    os.makedirs(OUT, exist_ok=True)
    n = B.N
    elev = np.load(os.path.join(TILES_DIR, "elevation_m.npy")).astype(np.float64)
    Z = np.fromfile(os.path.join(TILES_DIR, "viewer_heights.bin"), "<u2").reshape(n, n) / 100.0
    feat = np.fromfile(os.path.join(TILES_DIR, "viewer_features.bin"), np.uint8).reshape(n, n)
    lc = landcover_grid()
    for k, name in ((GLACIER, "glacier"), (FOREST, "forest"), (ROCK, "rock/scree"), (WATER, "water")):
        print(f"  {name:10s} {np.mean(lc == k):6.1%} of the model")
    np.save(os.path.join(TILES_DIR, "landcover.npy"), lc)
    G = grid_colors(elev, feat, lc)
    G.tofile(os.path.join(TILES_DIR, "viewer_colors.bin"))
    Image.fromarray(G).resize((1300, 1300), Image.LANCZOS).save(os.path.join(OUT, "colour_map_preview.png"))
    for d in ("final_stl", "final"):
        os.makedirs(os.path.join(RESULTS, d), exist_ok=True)
    for name in ("NW", "NE", "SW", "SE"):
        v, f = load_mesh(name)
        vertex_colors(v, name, Z, G).tofile(os.path.join(TILES_DIR, f"viewer_mesh_{name}_col.bin"))
        v, f = simplify(v, f)
        col = vertex_colors(v, name, Z, G)
        write_stl(os.path.join(RESULTS, "final_stl", f"courmayeur_{name}_26x26cm.stl"), v, f)
        obj = os.path.join(RESULTS, "final", f"courmayeur_{name}_26x26cm.obj")
        if os.path.exists(obj):
            os.remove(obj)                       # may be a hard link to the full-res tile
        write_obj(obj, v, f)
        p3 = os.path.join(OUT, f"courmayeur_{name}_26x26cm_color.3mf")
        pw = os.path.join(OUT, f"courmayeur_{name}_26x26cm_color.wrl")
        write_3mf(p3, v, f, col)
        write_wrl(pw, v, f, col)
        print(f"  {name}: {len(f):,} triangles; {os.path.getsize(p3) / 1e6:.0f} MB 3MF, {os.path.getsize(pw) / 1e6:.0f} MB VRML, "
              f"{len(np.unique(col, axis=0))} colours")


if __name__ == "__main__":
    main()
