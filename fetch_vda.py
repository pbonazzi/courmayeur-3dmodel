#!/usr/bin/env python3
"""Cut the model area out of the Aosta Valley 2 m lidar DTM without storing the whole thing.

The region publishes one ESRI ASCII grid (9.9 GB, Deflate64-zipped to 2.6 GB). The server
refuses byte ranges that deep into the file, so this streams it from the start in a single
request, inflates on the fly, keeps only the rows/columns over the model and stops as soon
as it has passed the model's southern edge. Result: data/dem_vda/vda_dtm_2m.npz for build_tiles.py.

Source: Regione Autonoma Valle d'Aosta, DTM 2005/2008 aggregato, quota ortometrica,
ED50 / UTM 32N (EPSG:23032), CC-BY. https://geoportale.regione.vda.it/download/dtm/
"""
import os, struct, sys, time, urllib.request
import inflate64
import numpy as np
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")          # downloaded inputs (fetch_data.py)
sys.path.insert(0, HERE)
from build_tiles import CX, CY, SIZE_M

URL = "https://geoprodotti.regione.vda.it/download/DTM0508_002/DTM0508_002_UNICO.zip"
MEMBER_OFFSET = 473369        # local header of DTM0508_002_UNICO.ASC inside the zip
MARGIN = 300                  # metres of extra ground around the model
OUT = os.path.join(DATA, "dem_vda", "vda_dtm_2m.npz")


def model_box_ed50():
    t = Transformer.from_crs(32632, 23032, always_xy=True)
    h = SIZE_M / 2 + MARGIN
    xs, ys = t.transform([CX - h, CX + h, CX - h, CX + h], [CY - h, CY - h, CY + h, CY + h])
    return min(xs), min(ys), max(xs), max(ys)


def lines(stream):
    inf, buf = inflate64.Inflater(), b""
    while True:
        chunk = stream.read(1 << 20)
        if not chunk:
            break
        buf += inf.inflate(chunk)
        *done, buf = buf.split(b"\n")
        yield from done
    if buf:
        yield buf


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    bx0, by0, bx1, by1 = model_box_ed50()
    r = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=300)
    skip = MEMBER_OFFSET
    while skip:
        skip -= len(r.read(min(skip, 1 << 16)))
    hdr = r.read(30)
    method, (nl, el) = struct.unpack("<H", hdr[8:10])[0], struct.unpack("<HH", hdr[26:30])
    assert method == 9, f"expected Deflate64, got method {method}"
    r.read(nl + el)

    it = lines(r)
    head = {}
    for _ in range(6):
        k, v = next(it).split()
        head[k.decode().lower()] = float(v.decode().replace(",", "."))
    ncols, nrows, res = int(head["ncols"]), int(head["nrows"]), head["cellsize"]
    gx0 = head["xllcorner"]
    gy0 = head["yllcorner"] + nrows * res                          # top edge
    c0, c1 = max(0, int((bx0 - gx0) // res)), min(ncols, int((bx1 - gx0) // res) + 1)
    r0, r1 = max(0, int((gy0 - by1) // res)), min(nrows, int((gy0 - by0) // res) + 1)
    print(f"Grid {ncols} x {nrows} @ {res} m; keeping rows {r0}-{r1}, cols {c0}-{c1}", flush=True)

    out = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
    t = time.time()
    for row, ln in enumerate(it):
        if row >= r1:
            break
        if row >= r0:
            v = np.array(ln.replace(b",", b".").split()[c0:c1], dtype=np.float32)   # decimal commas
            v[v <= -9000] = np.nan
            out[row - r0, :len(v)] = v
        if row % 1000 == 0:
            print(f"  row {row}/{r1}  {time.time() - t:.0f}s", flush=True)
    r.close()
    if row < r1 - 1:
        sys.exit(f"stream ended early at row {row}")
    np.savez_compressed(OUT, z=out, x0=gx0 + c0 * res, y0=gy0 - r0 * res, res=res)
    print(f"Saved {OUT}: {out.shape}, {np.isfinite(out).mean():.1%} valid, "
          f"{np.nanmin(out):.0f}-{np.nanmax(out):.0f} m")


if __name__ == "__main__":
    main()
