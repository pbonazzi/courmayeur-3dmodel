"""Text engraved into the front (south) side walls of the bottom tiles.

SW wall: the Christmas message. SE wall: title and data credits (Italian).
Glyph outlines come from the TrueType fonts (fontTools), are extruded with manifold3d and
subtracted ENGRAVE_MM deep into the wall, reading left to right when seen from the south.
"""
import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont

SERIF_BI = "/usr/share/fonts/truetype/noto/NotoSerif-BoldItalic.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

ENGRAVE_MM = 0.6
MARGIN_MM = 10.0          # keep clear of the tile corners
LINE_GAP = 0.35           # extra space between lines, in em

# per tile: lines from top to bottom as (text, font, em size in mm)
LABELS = {
    "SW": [
        ("Per mamma e papà, Buon Natale 2026 da Lucrezia e Pietro", SERIF_BI, 6.0),
    ],
    "SE": [
        ("Courmayeur e il massiccio del Monte Bianco · scala 1:38.500", SANS_B, 5.5),
        ("Dati: IGN LiDAR HD, RGE ALTI, Regione Autonoma Valle d'Aosta, INGV TINITALY, OpenStreetMap", SANS_B, 4.2),
    ],
}
BASELINE_MM = 2.5         # baseline of the lowest line, above the bottom of the tile


class _FlattenPen(BasePen):
    def __init__(self, glyphset, steps=6):
        super().__init__(glyphset)
        self.contours, self.cur, self.steps = [], [], steps

    def _moveTo(self, p):
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self.cur[-1]
        for t in np.linspace(0, 1, self.steps + 1)[1:]:
            self.cur.append(tuple((1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1)
                                  + 3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3)))

    def _qCurveToOne(self, p1, p2):
        p0 = self.cur[-1]
        for t in np.linspace(0, 1, self.steps + 1)[1:]:
            self.cur.append(tuple((1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)))

    def _closePath(self):
        if len(self.cur) > 2:
            self.contours.append(np.array(self.cur, float))
        self.cur = []

    _endPath = _closePath


_fonts = {}


def text_polygons(text, font_path, em_mm):
    """Outlines of a line of text in mm, baseline at y = 0, starting at x = 0. Returns (polys, width)."""
    if font_path not in _fonts:
        _fonts[font_path] = TTFont(font_path)
    font = _fonts[font_path]
    gs, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
    scale = em_mm / font["head"].unitsPerEm
    polys, x = [], 0.0
    for ch in text:
        name = cmap.get(ord(ch))
        if name is None:
            name = cmap[ord("?")]
        pen = _FlattenPen(gs)
        gs[name].draw(pen)
        for c in pen.contours:
            polys.append(np.column_stack([x + c[:, 0] * scale, c[:, 1] * scale]))
        x += hmtx[name][0] * scale
    return polys, x


def engrave(solid, tile_name, tile_size_mm):
    """Subtract this tile's text from its south wall (manifold3d Manifold in tile coordinates)."""
    import manifold3d as mf
    lines = LABELS.get(tile_name)
    if not lines:
        return solid
    usable = tile_size_mm - 2 * MARGIN_MM
    y = BASELINE_MM
    cutters = []
    for text, font, em in reversed(lines):                 # build from the bottom line up
        polys, w = text_polygons(text, font, em)
        if w > usable:                                       # shrink to fit the wall
            em = em * usable / w
            polys, w = text_polygons(text, font, em)
        x0 = (tile_size_mm - w) / 2
        cs = mf.CrossSection([p + [x0, y] for p in polys], mf.FillRule.NonZero)
        cutters.append(cs)
        y += em * (1 + LINE_GAP)
    text = mf.CrossSection.batch_union(cutters) if hasattr(mf.CrossSection, "batch_union") else sum(cutters[1:], cutters[0])
    depth = ENGRAVE_MM + 0.5
    cutter = text.extrude(depth)                             # z in [0, depth]
    # text x -> tile x, text y -> height, extrusion -> into the wall from just outside (y = -0.5)
    m = np.array([[1, 0, 0, 0], [0, 0, 1, -0.5], [0, 1, 0, 0]], float)
    cutter = cutter.transform(m)
    print(f"    engraved {len(lines)} lines on the {tile_name} south wall, top at {y:.1f} mm")
    return solid - cutter
