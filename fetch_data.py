#!/usr/bin/env python3
"""Download every input the build needs (about 1 GB). Files that already exist are skipped.

  data/dem_lidarhd/   IGN LiDAR HD terrain model, 4 x 10 km pieces at 2.5 m (IGN WMS, UTM 32N)
  data/dem_ign/       IGN RGE ALTI, same layout (fallback on the French side)
  data/dem_tinitaly/  INGV TINITALY 1.1 tile w50530 (10 m, Italy)
  data/dem_vda/       Regione Valle d'Aosta lidar DTM 2005/2008, 2 m, cut to the model (fetch_vda.py)
  osm/           OpenStreetMap: Mont Blanc Tunnel ways, roads + buildings, landcover
AWS Terrain Tiles (last-resort fill) are fetched on demand by build_tiles.py into data/tile_cache/.
"""
import io, json, os, sys, time, urllib.parse, urllib.request, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")          # downloaded inputs (fetch_data.py)
sys.path.insert(0, HERE)
from build_tiles import CX, CY

UA = {"User-Agent": "courmayeur-terrain-model/1.0 (personal project)"}
OSM_BBOX = "45.722,6.789,45.908,7.050"          # south, west, north, east: covers the model square
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]


def get(url, data=None, timeout=900, tries=4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"    retry after {e}")
            time.sleep(15 * (attempt + 1))


def ign_pieces(layer, folder, prefix):
    """Four 10 x 10 km pieces covering the model, 4000 px each (2.5 m), float32 BIL."""
    os.makedirs(os.path.join(DATA, folder), exist_ok=True)
    n = 4000
    for j, y0 in enumerate((CY - 10000, CY)):
        for i, x0 in enumerate((CX - 10000, CX)):
            p = os.path.join(DATA, folder, f"{prefix}_{i}{j}.f32")
            if os.path.exists(p):
                continue
            url = ("https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=" + layer +
                   f"&STYLES=&CRS=EPSG:32632&BBOX={x0},{y0},{x0 + 10000},{y0 + 10000}"
                   f"&WIDTH={n}&HEIGHT={n}&FORMAT=image/x-bil;bits=32")
            d = get(url)
            assert len(d) == n * n * 4, f"unexpected answer for {p}: {d[:200]!r}"
            open(p, "wb").write(d)
            print(f"  {os.path.relpath(p, HERE)}")


def tinitaly():
    out = os.path.join(DATA, "dem_tinitaly")
    if os.path.exists(os.path.join(out, "w50530_s10", "w50530_s10.tif")):
        return
    os.makedirs(out, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(get("https://tinitaly.pi.ingv.it/data_1.1/w50530_s10/w50530_s10.zip")))
    z.extractall(out)
    print("  data/dem_tinitaly/w50530_s10/")


def osm_tunnel_ways():
    from build_tunnel import WAYS
    os.makedirs(os.path.join(DATA, "osm"), exist_ok=True)
    for w in WAYS:
        p = os.path.join(DATA, "osm", f"way_{w}.json")
        if not os.path.exists(p):
            open(p, "wb").write(get(f"https://api.openstreetmap.org/api/0.6/way/{w}/full.json", timeout=120))
            print(f"  data/osm/way_{w}.json")
            time.sleep(1)


def overpass(name, query):
    p = os.path.join(DATA, "osm", name)
    if os.path.exists(p):
        return
    q = f"[out:json][timeout:240];({query});out tags geom;"
    for i, server in enumerate(OVERPASS * 2):
        try:
            d = get(server, data=urllib.parse.urlencode({"data": q}).encode(), timeout=400, tries=1)
            json.loads(d)
            open(p, "wb").write(d)
            print(f"  osm/{name}")
            return
        except Exception as e:
            print(f"    {server}: {e}")
            time.sleep(30)
    raise RuntimeError(f"Overpass failed for {name}; try again later")


def main():
    print("IGN LiDAR HD"); ign_pieces("IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", "dem_lidarhd", "lidarhd")
    print("IGN RGE ALTI"); ign_pieces("ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES", "dem_ign", "ign")
    print("TINITALY"); tinitaly()
    print("Valle d'Aosta lidar (streams ~1.3 GB, keeps ~230 MB)")
    if not os.path.exists(os.path.join(DATA, "dem_vda", "vda_dtm_2m.npz")):
        import fetch_vda
        fetch_vda.main()
    print("OpenStreetMap")
    osm_tunnel_ways()
    b = OSM_BBOX
    roads = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
             "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|service|track")
    overpass("features.json", f'way["building"]({b});way["highway"~"^({roads})$"]({b});')
    overpass("landcover.json",
             f'way["natural"~"^(glacier|wood|bare_rock|scree|water|shingle|rock)$"]({b});'
             f'relation["natural"~"^(glacier|wood|bare_rock|scree|water)$"]({b});'
             f'way["landuse"="forest"]({b});relation["landuse"="forest"]({b});way["waterway"="riverbank"]({b});')
    print("done")


if __name__ == "__main__":
    main()
