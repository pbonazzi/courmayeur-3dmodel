"""Roads, buildings and tunnel portals stamped into the model's height grid (millimetres).

Roads and buildings: OpenStreetMap (© OpenStreetMap contributors, ODbL), saved in data/osm/features.json.
At 1 : 38 500 a house is ~0.3 mm and a road ~0.2 mm, so both are exaggerated to print:
  roads      raised strips ROAD_WIDTH_MM wide, ROAD_RAISE_MM high, following the terrain
  buildings  flat-roofed blocks on their real footprint, BUILDING_EXAG x their real height
(The Mont Blanc Tunnel portals are opened by the bore itself: see build_tunnel.py.)
Also returns a feature mask (0 terrain, 1 road, 2 building, 3 highlighted building) for the viewer.
"""
import json, os
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")          # downloaded inputs (fetch_data.py)

ROAD_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential"}
ROAD_WIDTH_MM = 0.6
ROAD_RAISE_MM = 0.3
BUILDING_EXAG = 3.0
BUILDING_DEFAULT_M = 8.0          # when OSM has no height / levels
BUILDING_MIN_MM = 0.4

ROAD, BUILDING, HIGHLIGHT = 1, 2, 3
# OSM building (way) ids to show in gold in the viewer and colour files; geometry is unchanged
HIGHLIGHT_BUILDINGS = set()


def _building_height_m(tags):
    try:
        if "height" in tags:
            return float(str(tags["height"]).split()[0].replace(",", "."))
        if "building:levels" in tags:
            return 3.0 * float(tags["building:levels"]) + 2
    except ValueError:
        pass
    return BUILDING_DEFAULT_M


def apply(zmm, grid):
    """zmm: (N, N) heights in mm, row 0 = north. grid: dict with x0, y0 (UTM 32N of vertex [0,0]),
    step_m (ground metres per vertex), step_mm, mm_per_m. Modifies zmm in place; returns the mask."""
    n = zmm.shape[0]
    mask = np.zeros(zmm.shape, np.uint8)
    t = Transformer.from_crs(4326, 32632, always_xy=True)

    def to_px(geom):
        lon = np.array([p["lon"] for p in geom]); lat = np.array([p["lat"] for p in geom])
        x, y = t.transform(lon, lat)
        return list(zip((x - grid["x0"]) / grid["step_m"], (grid["y0"] - y) / grid["step_m"]))

    elements = json.load(open(os.path.join(DATA, "osm", "features.json")))["elements"]

    # roads
    img = Image.new("L", (n, n), 0); draw = ImageDraw.Draw(img)
    w = max(1, int(round(ROAD_WIDTH_MM / grid["step_mm"])))
    nroads = 0
    for e in elements:
        tg = e["tags"]
        if tg.get("highway") in ROAD_CLASSES and tg.get("tunnel", "no") == "no" and len(e.get("geometry", [])) > 1:
            draw.line(to_px(e["geometry"]), fill=1, width=w, joint="curve"); nroads += 1
    road = np.asarray(img, bool)

    # buildings: label raster so each footprint gets one flat roof
    lab_img = Image.new("I", (n, n), 0); draw = ImageDraw.Draw(lab_img)
    heights = [0.0]
    highlight_labels = []
    for e in elements:
        if "building" in e["tags"] and len(e.get("geometry", [])) > 2:
            heights.append(_building_height_m(e["tags"]))
            i = len(heights) - 1
            draw.polygon(to_px(e["geometry"]), fill=i, outline=i)
            if e["id"] in HIGHLIGHT_BUILDINGS:
                highlight_labels.append(i)
    labels = np.asarray(lab_img, np.int32)
    nb = len(heights) - 1
    idx = np.arange(1, nb + 1)
    roof_base = np.asarray(ndimage.maximum(zmm, labels, idx))
    h_mm = np.maximum(np.array(heights[1:]) * grid["mm_per_m"] * BUILDING_EXAG, BUILDING_MIN_MM)
    top = np.r_[0, roof_base + h_mm]
    bld = labels > 0

    zmm[road & ~bld] += ROAD_RAISE_MM
    zmm[bld] = top[labels[bld]]
    mask[road] = ROAD; mask[bld] = BUILDING
    mask[np.isin(labels, highlight_labels)] = HIGHLIGHT
    print(f"Stamped {nroads} roads and {nb} buildings")

    return mask
