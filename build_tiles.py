#!/usr/bin/env python3
"""High-resolution Courmayeur terrain model, split into 2 x 2 printable tiles.

Total footprint 520 x 520 mm (each tile 260 x 260 mm), 20 x 20 km of ground,
model frame = UTM 32N (WGS84), +Y = grid north.

Elevation sources, best first:
  1. IGN LiDAR HD MNT (France + a strip over the border), at 2.5 m -> data/dem_lidarhd/*.f32
  2. Regione Valle d'Aosta lidar DTM 2005/2008, 2 m (CC-BY)        -> data/dem_vda/*.npz
  2b. IGN RGE ALTI (France only), fetched at 2.5 m                 -> data/dem_ign/*.f32
  3. TINITALY 1.1 (INGV, Italy), 10 m (CC-BY 4.0)                  -> data/dem_tinitaly/
Roads, buildings and tunnel portals are stamped in afterwards by features.py (FEATURES=0 to skip).
  4. AWS Terrain Tiles (≈25 m) for anything still missing          -> data/tile_cache/
"""
import glob, math, os, sys
import numpy as np
from pyproj import Transformer
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")          # downloaded inputs (fetch_data.py)
RESULTS = os.path.join(HERE, "results")    # generated models and viewer data
sys.path.insert(0, HERE)

CX, CY = 338406, 5075597            # UTM 32N centre (45.815 N, 6.920 E)
SIZE_M = 20_000
MODEL_MM = 520.0
STEP_MM = 0.2                       # vertex spacing -> 7.7 m on the ground
VERT_EXAG = float(os.environ.get("VERT_EXAG", 1.0))
BASE_MM = 8.0
TILES = 2                           # 2 x 2
OUT = os.path.join(RESULTS, "tiles")

S = MODEL_MM / SIZE_M               # mm per metre
N = int(round(MODEL_MM / STEP_MM)) + 1           # 2601 vertices across the whole model
GSTEP = SIZE_M / (N - 1)                         # ground metres per vertex


def grid_xy():
    xs = CX - SIZE_M / 2 + np.arange(N) * GSTEP
    ys = CY + SIZE_M / 2 - np.arange(N) * GSTEP      # row 0 = north
    return np.meshgrid(xs, ys)


def sample(src, x0, y0, res, X, Y, smooth_m):
    """Bilinear sample of a north-up raster (top-left corner x0,y0, pixel res) at X,Y,
    after a Gaussian pre-filter so detail finer than the output grid doesn't alias."""
    valid = np.isfinite(src)
    sigma = max(0.0, 0.5 * smooth_m / res)
    if sigma > 0.3:
        filled = np.where(valid, src, 0.0)
        w = ndimage.gaussian_filter(valid.astype(np.float32), sigma)
        src = ndimage.gaussian_filter(filled, sigma) / np.maximum(w, 1e-6)
        valid = w > 0.5
    c = (X - x0) / res - 0.5
    r = (y0 - Y) / res - 0.5
    coords = np.array([r.ravel(), c.ravel()])
    z = ndimage.map_coordinates(np.where(valid, src, 0.0), coords, order=1, mode="nearest")
    ok = ndimage.map_coordinates(valid.astype(np.float32), coords, order=1, mode="nearest") > 0.999
    h, w = src.shape
    ok &= (coords[0] > -1) & (coords[0] < h) & (coords[1] > -1) & (coords[1] < w)
    z[~ok] = np.nan
    return z.reshape(X.shape)


def load_lidarhd(X, Y):
    """IGN LiDAR HD terrain model, 4 pieces of 10 km at 2.5 m in UTM 32N (fetched from the IGN WMS)."""
    n = 4000
    a = np.full((2 * n, 2 * n), np.nan, np.float32)
    for p in sorted(glob.glob(os.path.join(DATA, "dem_lidarhd", "lidarhd_*.f32"))):
        i, j = int(p[-6]), int(p[-5])
        a[(1 - j) * n:(2 - j) * n, i * n:(i + 1) * n] = np.fromfile(p, "<f4").reshape(n, n)
    bad = ~(a > 500)                                   # no-data (~ -999) and NaN
    bad = ndimage.binary_dilation(bad, iterations=3)   # drop pixels resampled across the no-data edge
    a[bad] = np.nan
    return sample(a, CX - 10000, CY + 10000, 10000 / n, X, Y, GSTEP)


def load_ign(X, Y):
    # stitch the 2 x 2 pieces first so their seams don't show
    n = 4000
    a = np.full((2 * n, 2 * n), np.nan, np.float32)
    for p in sorted(glob.glob(os.path.join(DATA, "dem_ign", "ign_*.f32"))):
        i, j = int(p[-6]), int(p[-5])                       # i: west/east, j: south/north
        a[(1 - j) * n:(2 - j) * n, i * n:(i + 1) * n] = np.fromfile(p, "<f4").reshape(n, n)
    a[a < -1000] = np.nan
    return sample(a, CX - 10000, CY + 10000, 10000 / n, X, Y, GSTEP)


def load_vda(X, Y):
    """VdA rasters are ED50 / UTM 32N (EPSG:23032): move the output grid into that frame."""
    files = sorted(glob.glob(os.path.join(DATA, "dem_vda", "*.npz")))
    out = np.full(X.shape, np.nan)
    if not files:
        return out
    t = Transformer.from_crs(32632, 23032, always_xy=True)
    Xe, Ye = t.transform(X, Y)
    for p in files:
        d = np.load(p)
        a, x0, y0, res = d["z"], float(d["x0"]), float(d["y0"]), float(d["res"])
        # 2 m is far finer than the 7.7 m print grid: average 2 x 2 blocks to save memory
        while res * 2 < GSTEP / 1.5:
            h2, w2 = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
            blk = a[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2)
            cnt = np.isfinite(blk).sum(axis=(1, 3))
            a = np.where(cnt > 0, np.nansum(blk, axis=(1, 3)) / np.maximum(cnt, 1), np.nan).astype(np.float32)
            res *= 2
        # only sample where this sheet overlaps
        h, w = a.shape
        m = (Xe >= x0) & (Xe <= x0 + w * res) & (Ye <= y0) & (Ye >= y0 - h * res) & np.isnan(out)
        if not m.any():
            continue
        rr, cc = np.where(m)
        sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
        z = sample(a, x0, y0, res, Xe[sl], Ye[sl], GSTEP)
        o = out[sl]
        out[sl] = np.where(np.isnan(o), z, o)
    return out


def load_tinitaly(X, Y):
    import tifffile
    out = np.full(X.shape, np.nan)
    for p in glob.glob(os.path.join(DATA, "dem_tinitaly", "*", "*.tif")):
        page = tifffile.TiffFile(p).pages[0]
        a = page.asarray().astype(np.float64)
        a[a <= -9000] = np.nan
        res = page.tags["ModelPixelScaleTag"].value[0]
        x0, y0 = page.tags["ModelTiepointTag"].value[3:5]          # EPSG:32632, top-left corner
        z = sample(a, x0, y0, res, X, Y, GSTEP)
        out = np.where(np.isnan(out), z, out)
    return out


def terrarium_tile(z, x, y):
    """AWS Terrain Tiles (Terrarium PNG), cached in data/tile_cache/."""
    import urllib.request
    from PIL import Image
    cache = os.path.join(DATA, "tile_cache")
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, f"{z}_{x}_{y}.png")
    if not os.path.exists(p):
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
        with urllib.request.urlopen(url, timeout=60) as r, open(p, "wb") as f:
            f.write(r.read())
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
    return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768


def load_terrarium(X, Y):
    t = Transformer.from_crs(32632, 4326, always_xy=True)
    lon, lat = t.transform(X, Y)
    z = 13
    px = (lon + 180) / 360 * 256 * 2 ** z
    py = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * 256 * 2 ** z
    tx0, ty0 = int(px.min() // 256), int(py.min() // 256)
    tx1, ty1 = int(px.max() // 256), int(py.max() // 256)
    mosaic = np.vstack([np.hstack([terrarium_tile(z, tx, ty) for tx in range(tx0, tx1 + 1)])
                        for ty in range(ty0, ty1 + 1)])
    coords = np.array([(py - ty0 * 256 - 0.5).ravel(), (px - tx0 * 256 - 0.5).ravel()])
    return ndimage.map_coordinates(mosaic, coords, order=1).reshape(X.shape)


def blend(primary, secondary, feather_px=20):
    """Keep primary untouched wherever it exists; fill its gaps from secondary, shifted by the
    primary-secondary difference found at the nearest primary pixel (fading out over feather_px),
    so the two meet without a step and the primary's edge pixels (often summits on the border)
    aren't diluted."""
    have = np.isfinite(primary)
    if have.all():
        return primary
    if not have.any():
        return secondary
    diff = np.where(have, primary - secondary, 0.0)
    w = ndimage.gaussian_filter(have.astype(np.float32), 2)
    diff = np.where(have, ndimage.gaussian_filter(diff, 2) / np.maximum(w, 1e-6), 0.0)
    dist, (ri, ci) = ndimage.distance_transform_edt(~have, return_indices=True)
    fade = np.clip(1 - dist / feather_px, 0, 1)
    shift = ndimage.gaussian_filter(diff[ri, ci] * fade, 3)    # nearest-pixel copy leaves streaks
    return np.where(have, primary, secondary + shift)


def write_tile(path, Xmm, Ymm, Z, cuts=()):
    """Closed solid from a height grid; each polyline in cuts is bored out as a round tunnel."""
    n, m = Z.shape
    idx = np.arange(n * m).reshape(n, m) + 1
    top = np.column_stack([Xmm.ravel(), Ymm.ravel(), Z.ravel()])
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    bot = top[ring - 1].copy(); bot[:, 2] = 0.0
    bot_ids = np.arange(len(ring)) + n * m + 1
    centre = n * m + len(ring) + 1
    verts = np.vstack([top, bot, [[Xmm.mean(), Ymm.mean(), 0.0]]])
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    r0, r1 = ring, np.roll(ring, -1)
    b0, b1 = bot_ids, np.roll(bot_ids, -1)
    F = np.vstack([np.column_stack([a, c, b]), np.column_stack([b, c, d]),
                   np.column_stack([r0, r1, b1]), np.column_stack([r0, b1, b0]),
                   np.column_stack([b0, b1, np.full_like(b0, centre)])])
    if len(cuts):
        verts, F = bore(verts, F - 1, cuts)
        F = F + 1
    if HOLLOW:
        verts, F = hollow(verts, F - 1, Z, Xmm[0, 1] - Xmm[0, 0])
        F = F + 1
    tile_name = os.path.basename(path).split("_")[1]
    import labels
    if labels.LABELS.get(tile_name):
        import manifold3d as mf
        solid = mf.Manifold(mf.Mesh(vert_properties=verts.astype(np.float32), tri_verts=(F - 1).astype(np.uint32)))
        solid = labels.engrave(solid, tile_name, MODEL_MM / TILES)
        assert solid.status() == mf.Error.NoError, solid.status()
        out = solid.to_mesh()
        verts = np.asarray(out.vert_properties)[:, :3].astype(np.float64)
        F = np.asarray(out.tri_verts).astype(np.int64) + 1
    with open(path, "w") as f:
        f.write(f"# Courmayeur terrain tile, millimetres, +Y = north\no {os.path.splitext(os.path.basename(path))[0]}\n")
        np.savetxt(f, verts, fmt="v %.3f %.3f %.3f")
        np.savetxt(f, F, fmt="f %d %d %d")
    print(f"  {os.path.basename(path)}: {len(verts):,} vertices, {len(F):,} faces, {os.path.getsize(path)/1e6:.0f} MB")
    return verts, F - 1


PATCH_HALF_MM = 4.0


def bore(verts, faces, cuts):
    """Subtract tunnel tubes from a closed mesh (manifold3d boolean). faces are 0-based."""
    import manifold3d as mf
    from build_tunnel import tube, TUBE_RADIUS_MM
    solid = mf.Manifold(mf.Mesh(vert_properties=verts.astype(np.float32), tri_verts=faces.astype(np.uint32)))
    for c in cuts:
        v, f = tube(c, TUBE_RADIUS_MM)
        solid = solid - mf.Manifold(mf.Mesh(vert_properties=v.astype(np.float32), tri_verts=f.astype(np.uint32)))
    assert solid.status() == mf.Error.NoError, solid.status()
    m = solid.to_mesh()
    return np.asarray(m.vert_properties)[:, :3].astype(np.float64), np.asarray(m.tri_verts).astype(np.int64)


HOLLOW = os.environ.get("HOLLOW", "1") != "0"
SHELL_MM = float(os.environ.get("SHELL_MM", 1.5))   # resin min wall 0.6-1 mm; 1.5 keeps 26 cm tiles stiff
CAVITY_STEP = 5            # cavity roof on a 1 mm grid (every 5th vertex)


def hollow(verts, faces, Z, step_mm):
    """Open-bottom shell: subtract a cavity whose roof stays >= SHELL_MM below the top surface
    (in 3D, via a disc-shaped minimum filter) and whose sides are SHELL_MM in from the tile edges."""
    import manifold3d as mf
    k = int(round(SHELL_MM / step_mm))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    roof = ndimage.minimum_filter(Z, footprint=(xx ** 2 + yy ** 2) <= k * k, mode="nearest") - SHELL_MM
    n = Z.shape[0]
    side = (n - 1) * step_mm
    idx = np.arange(0, n, CAVITY_STEP)
    if idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    r = roof[np.ix_(idx, idx)]
    xs = idx * step_mm
    keep = (xs >= SHELL_MM) & (xs <= side - SHELL_MM)
    xs = np.r_[SHELL_MM, xs[keep], side - SHELL_MM]
    # resample the roof onto the inset grid (nearest is fine at 1 mm)
    sel = np.r_[np.argmax(keep), np.where(keep)[0], len(keep) - 1 - np.argmax(keep[::-1])]
    r = r[np.ix_(sel, sel)]
    r = np.maximum(r, -0.5)                         # no cavity (it lies below the tile) where the shell is thin
    ys = side - xs                                  # row 0 = north
    X, Y = np.meshgrid(xs, ys)
    # closed solid from z = -1 (pokes out of the bottom, so the cavity is open) up to the roof
    m = len(xs)
    top = np.column_stack([X.ravel(), Y.ravel(), r.ravel()])
    bot = top.copy(); bot[:, 2] = -1.0
    V = np.vstack([top, bot])
    ids = np.arange(m * m).reshape(m, m)
    a, b_, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
    F = [np.column_stack([a, c, b_]), np.column_stack([b_, c, d]),
         np.column_stack([a + m * m, b_ + m * m, c + m * m]), np.column_stack([b_ + m * m, d + m * m, c + m * m])]
    ring = np.concatenate([ids[0, :], ids[1:, -1], ids[-1, -2::-1], ids[-2:0:-1, 0]])
    r0, r1 = ring, np.roll(ring, -1)
    F += [np.column_stack([r0, r1, r1 + m * m]), np.column_stack([r0, r1 + m * m, r0 + m * m])]
    cavity = mf.Manifold(mf.Mesh(vert_properties=V.astype(np.float32), tri_verts=np.vstack(F).astype(np.uint32)))
    assert cavity.status() == mf.Error.NoError, cavity.status()
    solid = mf.Manifold(mf.Mesh(vert_properties=verts.astype(np.float32), tri_verts=faces.astype(np.uint32)))
    before = solid.volume()
    solid = solid - cavity
    assert solid.status() == mf.Error.NoError, solid.status()
    out = solid.to_mesh()
    print(f"    hollowed: {before / 1000:.0f} -> {solid.volume() / 1000:.0f} cm3 ({SHELL_MM} mm shell, open bottom)")
    return np.asarray(out.vert_properties)[:, :3].astype(np.float64), np.asarray(out.tri_verts).astype(np.int64)


def _clip_poly(poly, axis, value, keep_less):
    """Sutherland-Hodgman: clip a 3D polygon against the plane coord[axis] = value."""
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        pin = p[axis] <= value if keep_less else p[axis] >= value
        qin = q[axis] <= value if keep_less else q[axis] >= value
        if pin:
            out.append(p)
        if pin != qin:
            t = (value - p[axis]) / (q[axis] - p[axis])
            out.append(p + t * (q - p))
    return out


def bore_patches(verts, faces, cuts, origin):
    """Pieces of the bored mesh around every place a tunnel piece ends (portal mouth or tile
    seam), clipped to a box, in global model mm, so the viewer can show the real holes."""
    out = []
    v3 = verts[faces]                                                    # (F, 3, 3)
    tmin, tmax = v3[..., :2].min(1), v3[..., :2].max(1)
    for c in cuts:
        for end in (c[0], c[-1]):
            lo = np.floor((end[:2] - PATCH_HALF_MM) / STEP_MM) * STEP_MM
            hi = np.ceil((end[:2] + PATCH_HALF_MM) / STEP_MM) * STEP_MM
            lo, hi = np.maximum(lo, 0), np.minimum(hi, MODEL_MM / TILES)
            cand = np.where(np.all(tmax >= lo - 1e-6, 1) & np.all(tmin <= hi + 1e-6, 1)
                            & ~np.all(v3[..., 2] < 1e-6, axis=1))[0]   # skip the flat bottom
            pv, pf = [], []
            for i in cand:
                poly = list(v3[i])
                for axis in (0, 1):
                    poly = _clip_poly(poly, axis, lo[axis], False)
                    if len(poly) < 3: break
                    poly = _clip_poly(poly, axis, hi[axis], True)
                    if len(poly) < 3: break
                if len(poly) < 3:
                    continue
                b = len(pv); pv += poly
                pf += [(b, b + k, b + k + 1) for k in range(1, len(poly) - 1)]
            pv = np.array(pv) + [origin[0], origin[1], 0]
            # merge duplicate corners so the viewer can smooth normals
            key = np.round(pv, 4)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            pf = inv.ravel()[np.array(pf)]
            pf = pf[(pf[:, 0] != pf[:, 1]) & (pf[:, 1] != pf[:, 2]) & (pf[:, 0] != pf[:, 2])]
            out.append({"box": [lo[0] + origin[0], lo[1] + origin[1], hi[0] + origin[0], hi[1] + origin[1]],
                        "verts": np.round(uniq, 3).ravel().tolist(), "faces": pf.ravel().tolist()})
    return out


LANDMARKS = {  # name: (lat, lon, real elevation m)
    "Mont Blanc": (45.8326, 6.8652, 4806), "Dent du Géant": (45.8622, 6.9517, 4013),
    "Grandes Jorasses": (45.8689, 6.9881, 4208), "Aiguille Noire": (45.8150, 6.8936, 3773),
    "Courmayeur": (45.7917, 6.9706, 1224), "Pré-Saint-Didier": (45.7647, 6.9853, 1004),
}


# Rock towers too slender for any of the DEMs (the Dent du Géant sits on the border, where
# the Aosta lidar stops and the French data is coarse). Modelled by hand as a tooth: a
# rounded-rectangle block with near-vertical walls and a domed crown with two summit points,
# dropping straight onto the measured ridge. Terrain is only raised, never lowered.
NEEDLES = [
    {"name": "Dent du Géant", "lat": 45.8622, "lon": 6.9517,
     "enabled": False,                  # IGN LiDAR HD measures the real tooth (3997 m at print resolution)
     "length_m": 60, "width_m": 36,     # block footprint
     "azimuth_deg": 70,                 # direction of the long axis (summit ridge), from north
     "wall_foot_m": 3850,               # where the vertical walls start
     "shoulder_m": 3990,                # top of the walls, where the crown begins
     "points": [(+0.4, 4013), (-0.4, 4009)],   # Pointe Graham, Pointe Sella: position along the axis (-1..1), height
     "pedestal_slope": None,            # no skirt: the walls drop straight to the measured ridge
     "snap_m": 120,                     # centre the tooth on the highest measured point within this radius
     "flank_slope": 1.6,                # where there is no lidar (French side), trim the coarse DEM dome so
     "flank_reach_m": 160},             # the ground falls away from the walls at this slope, fading out here
]


def tooth_height(u, v, nd):
    """Height of the tooth at local coords u (along the long axis) and v (across), metres."""
    a, b = nd["length_m"] / 2, nd["width_m"] / 2
    d = ((u / a) ** 4 + (v / b) ** 4) ** 0.25                  # 1 on the rounded-rectangle outline
    s = np.clip(u / a, -1, 1)
    crown = nd["shoulder_m"] + np.zeros_like(u)
    for pos, h in nd["points"]:                                 # two rounded summit bumps on a ridge
        crown = np.maximum(crown, h - (h - nd["shoulder_m"]) * ((s - pos) / 0.7) ** 2)
    crown -= (crown - nd["shoulder_m"]) * np.clip(d, 0, 1) ** 6 # roll the crown over at the edges
    if not nd.get("pedestal_slope"):
        return np.where(d <= 1, crown, -np.inf)
    outside = (d - 1) * min(a, b)                               # metres beyond the outline
    pedestal = nd["wall_foot_m"] - nd["pedestal_slope"] * outside
    return np.where(d <= 1, crown, pedestal)


def sculpt_needle(elev, X, Y, nd, measured=None):
    x, y = Transformer.from_crs(4326, 32632, always_xy=True).transform(nd["lon"], nd["lat"])
    # The published summit and the DEM's summit differ by tens of metres; build the tooth on the
    # DEM's own summit so it grows out of the measured peak instead of standing next to it.
    if nd.get("snap_m"):
        zone = np.hypot(X - x, Y - y) < nd["snap_m"]
        i = np.argmax(np.where(zone, elev, -np.inf))
        x, y = X.flat[i], Y.flat[i]
    reach = nd["length_m"] + ((nd["wall_foot_m"] - 3000) / nd["pedestal_slope"] if nd.get("pedestal_slope") else 0)
    near = np.hypot(X - x, Y - y) < min(reach, 400)
    az = np.radians(nd["azimuth_deg"])
    ax, ay = np.sin(az), np.cos(az)                             # unit vector of the long axis
    # supersample each vertex cell 5 x 5 and keep the highest, so the thin block can't fall
    # between grid points and its walls stay vertical (one cell = 7.7 m = 0.2 mm on the print)
    best = np.full(near.sum(), -np.inf)
    for oy in np.linspace(-0.4, 0.4, 5):
        for ox in np.linspace(-0.4, 0.4, 5):
            dx, dy = X[near] + ox * GSTEP - x, Y[near] + oy * GSTEP - y
            best = np.maximum(best, tooth_height(dx * ax + dy * ay, -dx * ay + dy * ax, nd))
    before = elev[near].max()
    if measured is not None and nd.get("flank_slope"):
        # distance (m) from the tooth's outline, from the same superellipse metric
        a, b = nd["length_m"] / 2, nd["width_m"] / 2
        dx, dy = X - x, Y - y
        u, v = dx * ax + dy * ay, -dx * ay + dy * ax
        out_m = np.maximum((((u / a) ** 4 + (v / b) ** 4) ** 0.25 - 1) * min(a, b), 0)
        reach = nd["flank_reach_m"]
        zone = (out_m < reach) & ~measured
        cap = nd["wall_foot_m"] - nd["flank_slope"] * out_m[zone]
        fade = np.clip((reach - out_m[zone]) / (0.4 * reach), 0, 1)   # full effect inside 60 % of reach
        cut = np.maximum(elev[zone] - cap, 0) * fade
        elev[zone] -= cut
        print(f"  trimmed the unmeasured flank of {nd['name']} by up to {cut.max():.0f} m")
    elev[near] = np.maximum(elev[near], best)
    print(f"Sculpted {nd['name']}: {before:.0f} m -> {elev[near].max():.0f} m")


def export_viewer(zmm, elev):
    """Heights (mm x 100, uint16, row 0 = north) + metadata for viewer/index.html."""
    import json
    np.round(zmm * 100).astype("<u2").tofile(os.path.join(OUT, "viewer_heights.bin"))
    t = Transformer.from_crs(4326, 32632, always_xy=True)
    marks = []
    for name, (lat, lon, real) in LANDMARKS.items():
        x, y = t.transform(lon, lat)
        xm, ym = (x - CX + SIZE_M / 2) * S, (y - CY + SIZE_M / 2) * S
        r, c = int(round((MODEL_MM - ym) / STEP_MM)), int(round(xm / STEP_MM))
        marks.append({"name": name, "x": round(xm, 2), "y": round(ym, 2), "z": round(float(zmm[r, c]), 2),
                      "elev": real, "peak": real > 3000})
    meta = {"n": N, "size_mm": MODEL_MM, "step_mm": STEP_MM, "tiles": TILES, "base_mm": BASE_MM,
            "mm_per_m": S * VERT_EXAG, "zmin_m": float(elev.min()), "vert_exag": VERT_EXAG, "landmarks": marks}
    with open(os.path.join(OUT, "viewer_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    X, Y = grid_xy()
    print(f"Grid {N} x {N}, {GSTEP:.2f} m per vertex")
    lhd = load_lidarhd(X, Y); print(f"IGN LiDAR HD covers {np.isfinite(lhd).mean():.1%}")
    vda = load_vda(X, Y); print(f"Aosta lidar covers {np.isfinite(vda).mean():.1%}")
    ign = load_ign(X, Y); print(f"IGN covers {np.isfinite(ign).mean():.1%}")
    tin = load_tinitaly(X, Y); print(f"TINITALY covers {np.isfinite(tin).mean():.1%}")
    terr = load_terrarium(X, Y)
    # IGN pads non-French ground with coarse fill, so only trust it where Italy's DTM has no data
    ign[np.isfinite(tin)] = np.nan
    base = blend(blend(tin, ign), terr)
    elev = blend(lhd, blend(vda, base))      # newest lidar first: it covers both faces of the border summits
    measured = np.isfinite(lhd) | np.isfinite(vda)
    for nd in NEEDLES:
        if nd.get("enabled", True):
            sculpt_needle(elev, X, Y, nd, measured=measured)
    np.save(os.path.join(OUT, "elevation_m.npy"), elev.astype(np.float32))
    print(f"Elevation {np.nanmin(elev):.0f} .. {np.nanmax(elev):.0f} m")

    zmm = (elev - elev.min()) * S * VERT_EXAG + BASE_MM
    print(f"Model height {zmm.max():.1f} mm (vertical x{VERT_EXAG})")
    z_bare = zmm.copy()
    if os.environ.get("FEATURES", "1") != "0":
        import features
        mask = features.apply(zmm, {"x0": CX - SIZE_M / 2, "y0": CY + SIZE_M / 2, "step_m": GSTEP,
                                    "step_mm": STEP_MM, "mm_per_m": S * VERT_EXAG,
                                    "zmin_m": float(elev.min()), "base_mm": BASE_MM})
    else:
        mask = np.zeros(zmm.shape, np.uint8)
    import build_tunnel
    to_utm = Transformer.from_crs(4326, 32632, always_xy=True)

    def model_xy(lon, lat):
        x, y = to_utm.transform(lon, lat)
        return (x - CX + SIZE_M / 2) * S, (y - CY + SIZE_M / 2) * S
    bores, bore_path, mouths = build_tunnel.prepare({"zmin_m": float(elev.min()), "mm_per_m": S * VERT_EXAG,
                                                     "base_mm": BASE_MM}, z_bare, STEP_MM, zmm, model_xy)
    build_tunnel.clear_portals(zmm, z_bare, mask, bore_path, STEP_MM)
    build_tunnel.carve_approaches(zmm, mask, mouths, STEP_MM, model_xy, z_bare)
    mask.tofile(os.path.join(OUT, "viewer_features.bin"))
    export_viewer(zmm, elev)
    Xmm = np.arange(N) * STEP_MM
    Ymm = MODEL_MM - np.arange(N) * STEP_MM
    k = (N - 1) // TILES
    names = {(0, 0): "NW", (0, 1): "NE", (1, 0): "SW", (1, 1): "SE"}
    patches = []
    for tr in range(TILES):
        for tc in range(TILES):
            rs, cs = slice(tr * k, tr * k + k + 1), slice(tc * k, tc * k + k + 1)
            gx, gy = np.meshgrid(Xmm[cs], Ymm[rs])
            gx, gy = gx - gx.min(), gy - gy.min()    # each tile's own origin at its SW corner
            cuts = bores.get(names[(tr, tc)], [])
            v, f = write_tile(os.path.join(OUT, f"courmayeur_{names[(tr, tc)]}_26x26cm.obj"), gx, gy, zmm[rs, cs],
                              cuts=cuts)
            name = names[(tr, tc)]
            with open(os.path.join(OUT, f"viewer_mesh_{name}.bin"), "wb") as fh:   # exact print mesh for the viewer
                np.array([len(v), len(f)], "<i4").tofile(fh)
                v.astype("<f4").tofile(fh); f.astype("<u4").tofile(fh)
            # results/final/, final_stl/, final_color/ (upload-ready, simplified) are written by export_color.py
            if cuts:
                patches += bore_patches(v, f, cuts, (tc * MODEL_MM / TILES, (TILES - 1 - tr) * MODEL_MM / TILES))

    import json
    with open(os.path.join(OUT, "viewer_bore_patches.json"), "w") as f:
        json.dump(patches, f)
    print(f"  {len(patches)} bore patches for the viewer")


if __name__ == "__main__":
    main()
