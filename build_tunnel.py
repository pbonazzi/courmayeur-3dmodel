#!/usr/bin/env python3
"""Mont Blanc road tunnel for the terrain model.

Route: OpenStreetMap ways 1212015047 + 4525912 (Tunnel du Mont Blanc, French portal section first)
and 205544546 (Traforo del Monte Bianco), saved in osm/. Vertical profile from the published portal/summit heights:
French portal 1274 m, highest point 1395.5 m at mid-length, Italian portal 1381 m.

The DEMs smooth the cuttings at both portals, so over the last kilometre the bore is eased
so that its floor meets the modelled access road (OSM route, raised road strip) just outside
each portal: the road runs level into the tunnel, and the bore (subtracted from the tiles in
build_tiles.py) opens the hole where it meets the hillside.

Writes
  results/tiles/viewer_tunnel.json                 bore path + portals for viewer/index.html
  results/tiles/tunnel_<tile>_26x26cm.obj          only with TUNNEL_OBJS=1: the bore as a separate tube per tile
"""
import json, os, sys
import numpy as np
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")          # downloaded inputs (fetch_data.py)
sys.path.insert(0, HERE)
from build_tiles import CX, CY, SIZE_M, MODEL_MM, TILES, OUT

WAYS = [1212015047, 4525912, 205544546]      # French portal section, French half, Italian half
PROFILE = [(0.0, 1274.0), (0.5, 1395.5), (1.0, 1381.0)]   # (fraction of length, metres)
TUBE_RADIUS_MM = float(os.environ.get("TUNNEL_RADIUS_MM", 1.0))   # real bore ≈ 0.12 mm: exaggerated
SIDES = 24
PORTAL_ROOF_MM = 0.6     # rock left above the bore at the portal
EASE_M = 1000            # length over which the bore is eased onto the modelled portal height
EXIT_MM = 0.5            # bore pokes this far out of the portal face, into the cutting
SEAM_OVERLAP_MM = 0.6    # pieces poke this far past a tile edge so the cut goes cleanly through the wall
APPROACH_GRADE = 0.3     # ramp of the access-road cutting (mm up per mm out): steep but legible
APPROACH_HALF_WIDTH_MM = 0.8
APPROACH_FLOOR_MM = 0.3  # road level above the bottom of the bore
ROAD_SAMPLE_MM = 1.0     # the road height is read this far out along the access road
WRITE_TUBE_OBJS = os.environ.get("TUNNEL_OBJS") == "1"   # separate tube OBJs (the tiles already have the bore)
APPROACH_RAMP = False    # True: cut the access road down into the bore instead of raising the bore
APPROACH_MAX_MM = 15.0


def route_latlon():
    pts = []
    for w in WAYS:
        d = json.load(open(os.path.join(DATA, "osm", f"way_{w}.json")))["elements"]
        nodes = {e["id"]: (e["lat"], e["lon"]) for e in d if e["type"] == "node"}
        g = [nodes[n] for n in next(e for e in d if e["type"] == "way")["nodes"]]
        if pts:
            assert pts[-1] == g[0], "ways don't join"
            g = g[1:]
        pts += g
    return pts


def route_model(meta):
    t = Transformer.from_crs(4326, 32632, always_xy=True)
    xy = np.array([t.transform(lon, lat) for lat, lon in route_latlon()])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    # densify to ~50 m so the profile is smooth
    sd = np.linspace(0, s[-1], int(s[-1] // 50) + 1)
    x, y = np.interp(sd, s, xy[:, 0]), np.interp(sd, s, xy[:, 1])
    f = sd / s[-1]
    z_m = np.interp(f, [p[0] for p in PROFILE], [p[1] for p in PROFILE])
    S = MODEL_MM / SIZE_M
    xm, ym = (x - CX + SIZE_M / 2) * S, (y - CY + SIZE_M / 2) * S
    zm = (z_m - meta["zmin_m"]) * meta["mm_per_m"] + meta["base_mm"]
    return np.column_stack([xm, ym, zm]), z_m, s[-1]


def tube(path, r):
    """Closed tube (with end caps) along a polyline, as (verts, faces 0-based)."""
    n = len(path)
    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    up = np.array([0, 0, 1.0])
    side = np.cross(tang, up); side /= np.linalg.norm(side, axis=1, keepdims=True)
    nrm = np.cross(side, tang)
    a = np.linspace(0, 2 * np.pi, SIDES, endpoint=False)
    ring = (np.cos(a)[None, :, None] * side[:, None, :] + np.sin(a)[None, :, None] * nrm[:, None, :]) * r
    verts = (path[:, None, :] + ring).reshape(-1, 3)
    faces = []
    for i in range(n - 1):
        for k in range(SIDES):
            a0, a1 = i * SIDES + k, i * SIDES + (k + 1) % SIDES
            b0, b1 = a0 + SIDES, a1 + SIDES
            faces += [(a0, b0, a1), (a1, b0, b1)]
    c0, c1 = len(verts), len(verts) + 1
    verts = np.vstack([verts, path[0], path[-1]])
    for k in range(SIDES):
        faces.append((c0, k, (k + 1) % SIDES))
        last = (n - 1) * SIDES
        faces.append((c1, last + (k + 1) % SIDES, last + k))
    return verts, np.array(faces)


def surface_at(z_grid, x_mm, y_mm, step_mm):
    """Bilinear height of the model surface (row 0 = north) at model mm coordinates."""
    n = z_grid.shape[0]
    c = x_mm / step_mm; r = (n - 1) - y_mm / step_mm
    c0, r0 = int(c), int(r); fc, fr = c - c0, r - r0
    q = z_grid[r0:r0 + 2, c0:c0 + 2]
    return (q[0, 0] * (1 - fc) * (1 - fr) + q[0, 1] * fc * (1 - fr) + q[1, 0] * (1 - fc) * fr + q[1, 1] * fc * fr)


def _point_along(line, dist):
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    return np.array([np.interp(dist, seg, line[:, 0]), np.interp(dist, seg, line[:, 1])])


def bore_path(meta, z_bare, step_mm, z_road=None, model_xy=None):
    """Tunnel centreline in model mm, fitted so its floor meets the access roads at the portals
    (or, without roads, so it sits PORTAL_ROOF_MM under the hillside), run out of both slopes."""
    path, z_m, length = route_model(meta)
    roads = approach_roads(model_xy) if model_xy is not None else None
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    ease_mm = EASE_M * MODEL_MM / SIZE_M
    r = TUBE_RADIUS_MM
    for end, dist in ((0, seg), (-1, seg[-1] - seg)):
        if roads is not None and len(roads[0 if end == 0 else 1]) > 1:
            road = roads[0 if end == 0 else 1]
            q = _point_along(road, ROAD_SAMPLE_MM)
            road_z = surface_at(z_road, q[0], q[1], step_mm)
            target = road_z - APPROACH_FLOOR_MM + r
            print(f"  portal {'FR' if end == 0 else 'IT'}: road at {road_z:.2f} mm, bore centre {target:.2f} mm, "
                  f"hillside {surface_at(z_bare, path[end, 0], path[end, 1], step_mm):.2f} mm")
        else:
            target = surface_at(z_bare, path[end, 0], path[end, 1], step_mm) - PORTAL_ROOF_MM - r
        path[:, 2] += (target - path[end, 2]) * np.clip(1 - dist / ease_mm, 0, 1)
    out = []
    for end, inner in ((0, 6), (-1, -7)):        # direction out of the mountain from ~300 m inside
        d = path[end, :2] - path[inner, :2]; d /= np.linalg.norm(d)
        ext = [np.r_[path[end, :2] + d * t, path[end, 2]] for t in np.linspace(EXIT_MM / 2, EXIT_MM, 2)]
        out.append(np.array(ext))
    full = np.vstack([out[0][::-1], path, out[1]])
    return full, path[0].copy(), path[-1].copy(), length


def split_at(path, axis, value):
    """Split a polyline where it crosses axis=value, inserting the exact crossing point."""
    out, cur = [], [path[0]]
    for p, q in zip(path[:-1], path[1:]):
        if (p[axis] - value) * (q[axis] - value) < 0:
            t = (value - p[axis]) / (q[axis] - p[axis])
            m = p + t * (q - p)
            cur.append(m); out.append(np.array(cur)); cur = [m]
        cur.append(q)
    out.append(np.array(cur))
    return out


TILE_NAMES = {(0, 0): "NW", (0, 1): "NE", (1, 0): "SW", (1, 1): "SE"}


def tile_pieces(path):
    """{tile name: [piece in that tile's own coordinates, ...]}, each piece overlapping tile edges."""
    k = MODEL_MM / TILES
    out = {}
    for piece in (p for q in split_at(path, 0, k) for p in split_at(q, 1, k)):
        if len(piece) < 2:
            continue
        piece = piece.copy()
        for end, nb in ((0, 1), (-1, -2)):           # push ends that sit on a seam a little further
            on_seam = np.any(np.isclose(piece[end, :2] % k, 0, atol=1e-6) | np.isclose(piece[end, :2] % k, k, atol=1e-6))
            if on_seam:
                d = piece[end] - piece[nb]; d /= np.linalg.norm(d)
                piece = np.insert(piece, len(piece) if end == -1 else 0, piece[end] + d * SEAM_OVERLAP_MM, axis=0)
        mid = piece[len(piece) // 2]
        tc, tr = int(mid[0] // k), int((MODEL_MM - mid[1]) // k)
        name = TILE_NAMES[(tr, tc)]
        out.setdefault(name, []).append(piece - [tc * k, (TILES - 1 - tr) * k, 0])
    return out


def prepare(meta, z_bare, step_mm, z_road=None, model_xy=None):
    """Compute the bore, write the viewer JSON and per-tile tube OBJs; return tile_pieces()."""
    path, mouth_fr, mouth_it, length = bore_path(meta, z_bare, step_mm, z_road, model_xy)
    with open(os.path.join(OUT, "viewer_tunnel.json"), "w") as f:
        json.dump({"name": "Mont Blanc Tunnel", "length_km": round(length / 1000, 2), "radius_mm": TUBE_RADIUS_MM,
                   "runout_points": 2,
                   "path": np.round(path, 3).tolist(),
                   "portals": [{"name": "Tunnel portal (France)", "elev": 1274, "xyz": np.round(mouth_fr, 2).tolist()},
                               {"name": "Tunnel portal (Italy)", "elev": 1381, "xyz": np.round(mouth_it, 2).tolist()}]}, f)
    pieces = tile_pieces(path)
    for f in os.listdir(OUT):
        if f.startswith("tunnel_") and f.endswith(".obj"):
            os.remove(os.path.join(OUT, f))
    for name, ps in pieces.items() if WRITE_TUBE_OBJS else ():
        with open(os.path.join(OUT, f"tunnel_{name}_26x26cm.obj"), "w") as fh:
            fh.write(f"# Mont Blanc Tunnel bore, tile {name}, millimetres, same frame as courmayeur_{name}_26x26cm.obj\n"
                     f"# radius {TUBE_RADIUS_MM} mm (exaggerated: the real bore would be ~0.12 mm)\n")
            base = 0
            for i, pc in enumerate(ps):
                v, fc = tube(pc, TUBE_RADIUS_MM)
                fh.write(f"o tunnel_{name}_{i}\n")
                np.savetxt(fh, v, fmt="v %.3f %.3f %.3f")
                np.savetxt(fh, fc + 1 + base, fmt="f %d %d %d")
                base += len(v)
    print(f"Tunnel bore {length/1000:.2f} km, radius {TUBE_RADIUS_MM} mm, in tiles {', '.join(sorted(pieces))}")
    return pieces, path, (mouth_fr, mouth_it)


def approach_roads(model_xy):
    """For each portal, the road leaving it (OSM, chained way by way), as model-mm polylines
    starting at the portal. model_xy(lon, lat) -> (x_mm, y_mm)."""
    elements = json.load(open(os.path.join(DATA, "osm", "features.json")))["elements"]
    ways = []
    for e in elements:
        tg = e["tags"]
        if tg.get("highway") in ("trunk", "trunk_link", "primary", "motorway") and tg.get("tunnel", "no") in ("no", "building_passage"):
            ways.append((e["id"], tg["highway"], np.array([model_xy(p["lon"], p["lat"]) for p in e["geometry"]])))
    pts = route_latlon()
    out = []
    for lat, lon in (pts[0], pts[-1]):
        cur = np.array(model_xy(lon, lat)); line = [cur]; used = set(); length = 0.0
        while length < APPROACH_MAX_MM:
            best = None
            for wid, kind, g in ways:
                if wid in used:
                    continue
                for gg in (g, g[::-1]):
                    gap = np.linalg.norm(gg[0] - cur)
                    if gap < 0.05 and (best is None or (kind == "trunk", -gap) > best[0]):   # prefer the main road
                        best = ((kind == "trunk", -gap), wid, gg)
            if best is None:
                break
            _, wid, gg = best
            used.add(wid); line += list(gg[1:]); cur = gg[-1]
            length += np.sum(np.linalg.norm(np.diff(gg, axis=0), axis=1))
        out.append(np.array(line))
    return out


def carve_approaches(zmm, mask, mouths, step_mm, model_xy, z_bare, road_value=1, building_value=2,
                     clear_mm=1.2, clear_len_mm=4.0):
    """Cut a ramp along each access road, from the floor of the bore at the mouth up to the
    road surface, so the road visibly drives into the tunnel."""
    n = zmm.shape[0]
    r = TUBE_RADIUS_MM
    for road, mouth in zip(approach_roads(model_xy), mouths):
        floor = mouth[2] - r + APPROACH_FLOOR_MM
        road = np.vstack([mouth[:2], road[1:]])                 # start exactly at the fitted mouth
        seg_s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(road, axis=0), axis=1))]
        keep = seg_s <= APPROACH_MAX_MM + 1
        road, seg_s = road[keep], seg_s[keep]
        lo = road.min(0) - 2; hi = road.max(0) + 2
        c0, c1 = max(int(lo[0] / step_mm), 0), min(int(hi[0] / step_mm) + 2, n)
        r0, r1 = max(int((n - 1) - hi[1] / step_mm), 0), min(int((n - 1) - lo[1] / step_mm) + 2, n)
        rr, cc = np.mgrid[r0:r1, c0:c1]
        px, py = cc * step_mm, ((n - 1) - rr) * step_mm
        dist = np.full(px.shape, np.inf); along = np.zeros(px.shape)
        for i, (p, q) in enumerate(zip(road[:-1], road[1:])):
            v = q - p; L2 = v @ v
            t_raw = ((px - p[0]) * v[0] + (py - p[1]) * v[1]) / L2
            if i == 0:
                behind = t_raw < 0                               # keep the portal face flat: nothing cut behind it
            t = np.clip(t_raw, 0, 1)
            d = np.hypot(px - p[0] - t * v[0], py - p[1] - t * v[1])
            better = d < dist
            dist[better] = d[better]; along[better] = seg_s[i] + t[better] * np.sqrt(L2)
        ramp = floor + APPROACH_GRADE * along
        sub, msub = zmm[r0:r1, c0:c1], mask[r0:r1, c0:c1]
        # keep the view into the portal open: no buildings right beside the start of the cutting
        bsub = z_bare[r0:r1, c0:c1]
        clear = (msub == building_value) & (dist <= APPROACH_HALF_WIDTH_MM + clear_mm) & (along < clear_len_mm)
        sub[clear] = bsub[clear]; msub[clear] = 0
        cut = (dist <= APPROACH_HALF_WIDTH_MM) & (ramp < sub) & ~behind & APPROACH_RAMP
        sub[cut] = ramp[cut]
        msub[cut] = road_value
        if APPROACH_RAMP:
            print(f"  approach cutting: {np.count_nonzero(cut)} cells, ramp {along[cut].max() if cut.any() else 0:.1f} mm long")


def clear_portals(zmm, z_bare, mask, path, step_mm, margin_mm=0.6):
    """Remove buildings in front of and around both mouths so they can't plug the holes
    (the road strip stays, so it runs on into the tunnel)."""
    n = zmm.shape[0]
    run = 3
    for seg in (path[:run + 4], path[-(run + 4):]):         # run-out plus ~1 mm inside the mouth
        lo, hi = seg[:, :2].min(0) - TUBE_RADIUS_MM - margin_mm, seg[:, :2].max(0) + TUBE_RADIUS_MM + margin_mm
        c0, c1 = int(lo[0] / step_mm), int(hi[0] / step_mm) + 2
        r0, r1 = int((n - 1) - hi[1] / step_mm), int((n - 1) - lo[1] / step_mm) + 2
        rr, cc = np.mgrid[r0:r1, c0:c1]
        px, py = cc * step_mm, ((n - 1) - rr) * step_mm
        d = np.full(px.shape, np.inf)
        for p, q in zip(seg[:-1], seg[1:]):
            v = q[:2] - p[:2]; t = np.clip(((px - p[0]) * v[0] + (py - p[1]) * v[1]) / (v @ v), 0, 1)
            d = np.minimum(d, np.hypot(px - p[0] - t * v[0], py - p[1] - t * v[1]))
        near = (d < TUBE_RADIUS_MM + margin_mm) & (mask[r0:r1, c0:c1] == 2)       # buildings only
        zmm[r0:r1, c0:c1][near] = z_bare[r0:r1, c0:c1][near]
        mask[r0:r1, c0:c1][near] = 0


if __name__ == "__main__":
    import build_tiles as bt_
    meta = json.load(open(os.path.join(OUT, "viewer_meta.json")))
    e = np.load(os.path.join(OUT, "elevation_m.npy")).astype(float)
    prepare(meta, (e - e.min()) * meta["mm_per_m"] + meta["base_mm"], bt_.STEP_MM)
