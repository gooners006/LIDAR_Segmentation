"""Generate the D3 scaffold (.drawio) + a rough SVG preview of its geometry.
Content the user must supply is left as '?' placeholders."""
import sys
from html import escape
from xml.sax.saxutils import quoteattr

out_drawio, out_svg = sys.argv[1], sys.argv[2]
FONT = "fontFamily=Segoe UI;fontColor=#1F2933;"
ST = {
    "classical": "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#DCE8F5;strokeColor=#2B5C8A;strokeWidth=2;fontSize=19;" + FONT,
    "learned": "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#FDE6CC;strokeColor=#D9822B;strokeWidth=2;fontSize=19;" + FONT,
    "data": "ellipse;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#9AA0A6;strokeWidth=2;fontSize=19;" + FONT,
    "group": "rounded=1;arcSize=5;html=1;fillColor=none;strokeColor=#9AA0A6;strokeWidth=2;dashed=1;dashPattern=8 6;",
    "grouplabel": "text;html=1;align=left;verticalAlign=middle;fontSize=20;fontStyle=1;labelBackgroundColor=#FFFFFF;fontFamily=Segoe UI;fontColor=#3E4C59;",
    "outlabel": "text;html=1;align=center;verticalAlign=middle;fontSize=19;fontStyle=2;fontFamily=Segoe UI;fontColor=#6B7280;",
    "trainlabel": "text;html=1;align=center;verticalAlign=middle;fontSize=19;fontStyle=1;fontFamily=Segoe UI;fontColor=#D9822B;",
    "stage": "text;html=1;horizontal=0;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;" + FONT,
    "legendtext": "text;html=1;align=left;verticalAlign=middle;fontSize=19;fontFamily=Segoe UI;fontColor=#3E4C59;",
}
EDGE = "endArrow=block;endFill=1;html=1;strokeColor=#3E4C59;strokeWidth=3;"
BOX = '<b style="font-size:22px">?</b><br>?'   # name (bold) / method
DATA2 = '?<br><i>?</i>'

# id, style, x, y, w, h, value
V = [
    ("stage1", "stage", 5, 50, 30, 140, "DETECTION"),
    ("raw", "data", 50, 80, 120, 80, "Raw LiDAR scan"),
    ("g1", "group", 190, 50, 600, 140, ""),
    ("b1", "classical", 205, 65, 170, 110, BOX),
    ("b2", "classical", 405, 65, 170, 110, BOX),
    ("b3", "classical", 605, 65, 170, 110, BOX),
    ("g2", "group", 830, 50, 600, 140, ""),
    ("b4", "classical", 845, 65, 170, 110, BOX),
    ("b5", "learned", 1045, 65, 170, 110, BOX),
    ("b6", "classical", 1245, 65, 170, 110, BOX),
    ("det", "data", 1460, 80, 200, 80, DATA2),
    ("stage2", "stage", 5, 320, 30, 140, "COMPLETION"),
    ("g3", "group", 190, 320, 600, 140, ""),
    ("b7", "classical", 205, 335, 570, 110, BOX),
    ("g4", "group", 830, 320, 200, 140, ""),
    ("b8", "learned", 845, 335, 170, 110, BOX),
    ("done", "data", 1070, 350, 200, 80, DATA2),
    # phase labels on the dashed top edges
    ("gl1", "grouplabel", 200, 38, 330, 24, "phase ?"),
    ("gl2", "grouplabel", 840, 38, 290, 24, "phase ?"),
    ("gl3", "grouplabel", 200, 308, 230, 24, "phase ?"),
    ("gl4", "grouplabel", 840, 308, 120, 24, "phase ?"),
    # what flows out of each step
    ("o1", "outlabel", 205, 196, 170, 24, "→ ?"),
    ("o2", "outlabel", 405, 196, 170, 24, "→ ?"),
    ("o3", "outlabel", 605, 196, 170, 24, "→ ?"),
    ("o4", "outlabel", 845, 196, 170, 24, "→ ?"),
    ("o5", "outlabel", 1045, 196, 170, 24, "→ ?"),
    ("o6", "outlabel", 1245, 196, 170, 24, "→ ?"),
    ("o7", "outlabel", 205, 466, 570, 24, "→ ?"),
    ("o8", "outlabel", 830, 466, 200, 24, "→ ?"),
    # what trains the two learned steps
    ("t5", "trainlabel", 1070, 6, 200, 24, "trained on: ?"),
    ("t8", "trainlabel", 890, 276, 200, 24, "trained on: ?"),
    # legend
    ("lc", "classical", 1330, 372, 24, 24, ""),
    ("lct", "legendtext", 1362, 372, 230, 24, "classical (training-free)"),
    ("ll", "learned", 1330, 408, 24, 24, ""),
    ("llt", "legendtext", 1362, 408, 230, 24, "learned"),
]
# id, source, target, extra style, waypoints
E = [
    ("e0", "raw", "b1", "", []), ("e1", "b1", "b2", "", []), ("e2", "b2", "b3", "", []),
    ("e3", "b3", "b4", "", []), ("e4", "b4", "b5", "", []), ("e5", "b5", "b6", "", []),
    ("e6", "b6", "det", "", []),
    ("hand", "det", "b7",
     "edgeStyle=orthogonalEdgeStyle;rounded=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;",
     [(1560, 250), (120, 250), (120, 390)]),
    ("e7", "b7", "b8", "", []), ("e8", "b8", "done", "", []),
    ("tr5", "t5", "b5", "strokeColor=#D9822B;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.735;entryY=0;entryDx=0;entryDy=0;", []),
    ("tr8", "t8", "b8", "strokeColor=#D9822B;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.853;entryY=0;entryDx=0;entryDy=0;", []),
]

cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
for cid, st, x, y, w, h, val in V:
    cells.append(f'<mxCell id="{cid}" value={quoteattr(val)} style={quoteattr(ST[st])} vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')
for cid, s, t, extra, pts in E:
    style = EDGE + extra
    geo = '<mxGeometry relative="1" as="geometry">'
    if pts:
        geo += '<Array as="points">' + "".join(f'<mxPoint x="{px}" y="{py}" />' for px, py in pts) + "</Array>"
    geo += "</mxGeometry>"
    cells.append(f'<mxCell id="{cid}" style={quoteattr(style)} edge="1" parent="1" source="{s}" target="{t}">{geo}</mxCell>')

xml = ('<mxfile host="app.diagrams.net">\n  <diagram name="D3 master pipeline" id="d3">\n'
       '    <mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
       'page="1" pageScale="1" pageWidth="1680" pageHeight="560" math="0" shadow="0">\n      <root>\n        '
       + "\n        ".join(cells) + "\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n")
open(out_drawio, "w", encoding="utf-8").write(xml)

# ---- rough preview (geometry check only; draw.io's own rendering will differ slightly)
geom = {cid: (x, y, w, h) for cid, _, x, y, w, h, _ in V}
svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1680 500" width="1680" height="500" '
       'font-family="Segoe UI"><rect width="1680" height="500" fill="#fff"/>',
       '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
       '<path d="M0,0L10,5L0,10z" fill="#3E4C59"/></marker></defs>']
col = {"classical": ("#DCE8F5", "#2B5C8A"), "learned": ("#FDE6CC", "#D9822B"), "data": ("#F3F4F6", "#9AA0A6")}
for cid, st, x, y, w, h, val in V:
    if st == "group":
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="none" stroke="#9AA0A6" stroke-width="2" stroke-dasharray="8 6"/>')
for cid, s, t, extra, pts in E:
    sx, sy, sw, sh = geom[s]; tx, ty, tw, th = geom[t]
    if pts:
        p = [(sx + sw / 2, sy + sh)] + pts + [(tx, ty + th / 2)]
    elif cid.startswith("tr"):
        ex = float(extra.split("entryX=")[1].split(";")[0])
        p = [(sx + sw / 2, sy + sh), (tx + ex * tw, ty)]
    else:
        p = [(sx + sw, sy + sh / 2), (tx, ty + th / 2)]
    svg.append(f'<polyline points="{" ".join(f"{a},{b}" for a, b in p)}" fill="none" stroke="#3E4C59" stroke-width="3" marker-end="url(#a)"/>')
for cid, st, x, y, w, h, val in V:
    txt = val.replace('<b style="font-size:22px">', "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
    lines = txt.split("<br>")
    if st in col:
        f, s = col[st]
        shape = (f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}"' if st == "data"
                 else f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10"')
        svg.append(f'{shape} fill="{f}" stroke="{s}" stroke-width="2"/>')
    if st == "group" or not txt:
        continue
    if st == "stage":
        svg.append(f'<text transform="translate({x+w/2+7},{y+h/2}) rotate(-90)" font-size="22" font-weight="700" text-anchor="middle">{escape(txt)}</text>')
        continue
    anchor, tx_ = ("start", x) if st in ("grouplabel", "legendtext") else ("middle", x + w / 2)
    if st == "grouplabel":
        svg.append(f'<rect x="{x-4}" y="{y}" width="{len(txt)*11+8}" height="{h}" fill="#fff"/>')
    size = {"grouplabel": 20, "stage": 22}.get(st, 19)
    y0 = y + h / 2 - (len(lines) - 1) * 13 + 7
    for i, ln in enumerate(lines):
        svg.append(f'<text x="{tx_}" y="{y0 + i*26}" font-size="{size}" text-anchor="{anchor}">{escape(ln)}</text>')
svg.append("</svg>")
open(out_svg, "w", encoding="utf-8").write("\n".join(svg))
print("ok", len(V), "vertices", len(E), "edges")
