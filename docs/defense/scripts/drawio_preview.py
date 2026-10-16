"""Approximate renderer: .drawio (uncompressed XML) -> SVG. Geometry, colours, dashes, text with
naive word-wrap, straight / waypoint edges. Not draw.io's renderer; for layout checks only."""
import re, sys
import xml.etree.ElementTree as ET
from html import escape, unescape

src, out = sys.argv[1], sys.argv[2]
root = ET.parse(src).getroot()
cells = list(root.iter("mxCell"))


def sty(s):
    d = {}
    for part in (s or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1); d[k] = v
        elif part:
            d[part] = "1"
    return d


def lines_of(val):
    """html label -> list of (text, bold, italic, size or None)"""
    v = re.sub(r"<div>", "<br>", val or ""); v = re.sub(r"</div>", "", v)
    res = []
    for seg in v.split("<br>"):
        size = re.search(r"font-size:\s*(\d+)px", seg)
        bold = "<b>" in seg; ital = "<i>" in seg
        t = unescape(re.sub(r"<[^>]+>", "", seg)).strip()
        if t:
            res.append((t, bold, ital, int(size.group(1)) if size else None))
    return res


def wrap(t, width, size):
    cw = size * 0.53  # avg Segoe UI glyph width
    maxc = max(1, int(width / cw))
    out, cur = [], ""
    for w in t.split():
        if len(cur) + len(w) + (1 if cur else 0) <= maxc:
            cur = (cur + " " + w).strip()
        else:
            if cur: out.append(cur)
            cur = w
    if cur: out.append(cur)
    return out


V = {}
for c in cells:
    g = c.find("mxGeometry")
    if c.get("vertex") == "1" and g is not None:
        V[c.get("id")] = (float(g.get("x", 0)), float(g.get("y", 0)), float(g.get("width", 0)),
                          float(g.get("height", 0)), sty(c.get("style")), c.get("value", ""))
xs = [x + w for x, y, w, h, *_ in V.values()]; ys = [y + h for x, y, w, h, *_ in V.values()]
W, H = max(xs) + 20, max(ys) + 20
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI">',
       f'<rect width="{W}" height="{H}" fill="#fff"/>',
       '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
       '<path d="M0,0L10,5L0,10z" fill="context-stroke"/></marker></defs>']

# groups first (back), then edges, then shapes, then text
order = sorted(V.items(), key=lambda kv: 0 if kv[1][4].get("fillColor") == "none" else 1)
for cid, (x, y, w, h, s, val) in order:
    if s.get("fillColor") == "none":
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="none" stroke="{s.get("strokeColor","#000")}" '
                   f'stroke-width="{s.get("strokeWidth",1)}" stroke-dasharray="{s.get("dashPattern","6 4")}"/>')
for c in cells:
    if c.get("edge") != "1": continue
    s = sty(c.get("style")); a, b = V.get(c.get("source")), V.get(c.get("target"))
    if not a or not b: continue
    ax, ay, aw, ah = a[:4]; bx, by, bw, bh = b[:4]
    p0 = (ax + aw * float(s.get("exitX", 1)), ay + ah * float(s.get("exitY", 0.5)))
    p1 = (bx + bw * float(s.get("entryX", 0)), by + bh * float(s.get("entryY", 0.5)))
    pts = [(float(p.get("x")), float(p.get("y"))) for p in c.iter("mxPoint") if p.get("as") is None]
    poly = " ".join(f"{px},{py}" for px, py in [p0] + pts + [p1])
    svg.append(f'<polyline points="{poly}" fill="none" stroke="{s.get("strokeColor","#000")}" '
               f'stroke-width="{s.get("strokeWidth",1)}" marker-end="url(#a)"/>')
for cid, (x, y, w, h, s, val) in V.items():
    fill = s.get("fillColor", "#fff")
    if fill != "none" and "text" not in s:
        stroke = s.get("strokeColor", "#000"); sw = s.get("strokeWidth", 1)
        if "ellipse" in s:
            svg.append(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        else:
            svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    ls = lines_of(val)
    if not ls: continue
    base = float(s.get("fontSize", 12)); col = s.get("fontColor", "#000")
    fb = s.get("fontStyle", "0"); sbold = int(fb) & 1; sital = int(fb) & 2
    if s.get("horizontal") == "0":
        t = ls[0][0]
        svg.append(f'<text transform="translate({x+w/2+base/3},{y+h/2}) rotate(-90)" font-size="{base}" '
                   f'font-weight="700" fill="{col}" text-anchor="middle">{escape(t)}</text>')
        continue
    pad = 6 if "ellipse" not in s else w * 0.15
    rows = []
    for t, b_, i_, sz in ls:
        size = sz or base
        for r in (wrap(t, w - 2 * pad, size) if "whiteSpace" in s or "text" not in s else [t]):
            rows.append((r, b_ or sbold, i_ or sital, size))
    total = sum(r[3] * 1.2 for r in rows)
    cy = y + h / 2 - total / 2
    align = s.get("align", "center")
    anchor, tx = {"left": ("start", x), "right": ("end", x + w)}.get(align, ("middle", x + w / 2))
    for r, b_, i_, size in rows:
        cy += size * 1.2
        if s.get("labelBackgroundColor"):
            tw = len(r) * size * (0.56 if (b_ or sbold) else 0.5)
            bx = tx if anchor == "start" else tx - tw / 2
            # tight to the glyphs: cover the dashed line behind the text, not the boxes below
            svg.append(f'<rect x="{bx-4}" y="{cy-size*1.05}" width="{tw+8}" height="{size*1.05}" fill="{s["labelBackgroundColor"]}"/>')
        extra = (' font-weight="700"' if b_ else "") + (' font-style="italic"' if i_ else "")
        svg.append(f'<text x="{tx}" y="{cy - size*0.25}" font-size="{size}" fill="{col}" '
                   f'text-anchor="{anchor}"{extra}>{escape(r)}</text>')
svg.append("</svg>")
open(out, "w", encoding="utf-8").write("\n".join(svg))
print(f"{len(V)} vertices -> {out} ({W:.0f}x{H:.0f})")
