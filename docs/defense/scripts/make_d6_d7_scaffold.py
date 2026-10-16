"""Generate the D6 (label mining flow) and D7 (simplified classifier) draw.io scaffolds.
Content the user must supply is left as '?' placeholders. Same styles as D3."""
import sys
from xml.sax.saxutils import quoteattr

out_dir = sys.argv[1]
FONT = "fontFamily=Segoe UI;fontColor=#1F2933;"
ST = {
    "classical": "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#DCE8F5;strokeColor=#2B5C8A;strokeWidth=2;fontSize=19;" + FONT,
    "learned": "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#FDE6CC;strokeColor=#D9822B;strokeWidth=2;fontSize=19;" + FONT,
    "data": "ellipse;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#9AA0A6;strokeWidth=2;fontSize=19;" + FONT,
    "decision": "rhombus;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#3E4C59;strokeWidth=2;fontSize=19;" + FONT,
    "group": "rounded=1;arcSize=5;html=1;fillColor=none;strokeColor=#9AA0A6;strokeWidth=2;dashed=1;dashPattern=8 6;",
    "grouplabel": "text;html=1;align=left;verticalAlign=middle;fontSize=20;fontStyle=1;labelBackgroundColor=#FFFFFF;fontFamily=Segoe UI;fontColor=#3E4C59;",
    "outlabel": "text;html=1;align=center;verticalAlign=middle;fontSize=19;fontStyle=2;fontFamily=Segoe UI;fontColor=#6B7280;",
    "keylabel": "text;html=1;align=center;verticalAlign=middle;fontSize=19;fontStyle=1;fontFamily=Segoe UI;fontColor=#D9822B;",
    "legendtext": "text;html=1;align=left;verticalAlign=middle;fontSize=19;fontFamily=Segoe UI;fontColor=#3E4C59;",
}
EDGE = "endArrow=block;endFill=1;html=1;strokeColor=#3E4C59;strokeWidth=3;fontSize=18;fontFamily=Segoe UI;fontStyle=1;labelBackgroundColor=#FFFFFF;"
ORTHO = "edgeStyle=orthogonalEdgeStyle;rounded=1;"
BOX = '<b style="font-size:21px">?</b><br>?'
DATA2 = '?<br><i>?</i>'
LEGEND = [
    ("lc", "classical", 0, 0, 24, 24, ""), ("lct", "legendtext", 32, 0, 230, 24, "classical (training-free)"),
    ("ll", "learned", 0, 34, 24, 24, ""), ("llt", "legendtext", 32, 34, 230, 24, "learned"),
]


def write(name, title, V, E, w, h, legend_xy):
    lx, ly = legend_xy
    V = V + [(c, s, x + lx, y + ly, ww, hh, v) for c, s, x, y, ww, hh, v in LEGEND]
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
    for cid, st, x, y, ww, hh, val in V:
        cells.append(f'<mxCell id="{cid}" value={quoteattr(val)} style={quoteattr(ST[st])} vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{ww}" height="{hh}" as="geometry" /></mxCell>')
    for cid, s, t, extra, label in E:
        cells.append(f'<mxCell id="{cid}" value={quoteattr(label)} style={quoteattr(EDGE + extra)} edge="1" '
                     f'parent="1" source="{s}" target="{t}"><mxGeometry relative="1" as="geometry" /></mxCell>')
    xml = (f'<mxfile host="app.diagrams.net">\n  <diagram name="{title}" id="{name}">\n'
           '    <mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
           f'page="1" pageScale="1" pageWidth="{w}" pageHeight="{h}" math="0" shadow="0">\n      <root>\n        '
           + "\n        ".join(cells) + "\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n")
    path = f"{out_dir}/{name}.drawio"
    open(path, "w", encoding="utf-8").write(xml)
    print("wrote", path, len(V), "vertices", len(E), "edges")


# ---------------- D7: simplified classifier (two branches)
V7 = [
    ("cl", "data", 20, 165, 160, 90, "one cluster"),
    ("g1", "group", 225, 30, 965, 150, ""),
    ("g2", "group", 225, 245, 965, 150, ""),
    ("gl1", "grouplabel", 235, 18, 220, 24, "? branch"),
    ("gl2", "grouplabel", 235, 233, 220, 24, "? branch"),
    ("b1", "classical", 245, 50, 210, 110, BOX),
    ("b2", "learned", 490, 50, 230, 110, BOX),
    ("b3", "classical", 755, 50, 170, 110, BOX),
    ("d1", "data", 960, 65, 210, 80, DATA2),
    ("b4", "classical", 245, 265, 210, 110, BOX),
    ("b5", "learned", 490, 265, 230, 110, BOX),
    ("d2", "data", 960, 280, 210, 80, DATA2),
    ("b6", "learned", 1230, 155, 190, 110, BOX),
    ("out", "data", 1460, 170, 200, 80, "car / not-car"),
    ("why1", "keylabel", 610, 196, 360, 28, "order doesn't matter because: ?"),
]
E7 = [
    ("e1", "cl", "b1", ORTHO + "exitX=1;exitY=0.25;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;", ""),
    ("e2", "cl", "b4", ORTHO + "exitX=1;exitY=0.75;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;", ""),
    ("e3", "b1", "b2", "", ""), ("e4", "b2", "b3", "", ""), ("e5", "b3", "d1", "", ""),
    ("e6", "b4", "b5", "", ""), ("e7", "b5", "d2", "", ""),
    ("e8", "d1", "b6", ORTHO + "entryX=0;entryY=0.25;entryDx=0;entryDy=0;", ""),
    ("e9", "d2", "b6", ORTHO + "entryX=0;entryY=0.75;entryDx=0;entryDy=0;", ""),
    ("e10", "b6", "out", "", ""),
]
write("d7_classifier", "D7 simplified classifier", V7, E7, 1680, 420, (1440, 320))

# ---------------- D6: auto-mined label flow
V6 = [
    ("keyA", "keylabel", 10, 40, 220, 28, "labels from: ?"),
    ("in", "data", 20, 150, 200, 110, DATA2),
    ("keyB", "keylabel", 250, 40, 700, 28, "clusters from: ?"),
    ("g1", "group", 245, 80, 700, 250, ""),
    ("b1", "classical", 265, 145, 200, 120, BOX),
    ("b2", "classical", 500, 145, 200, 120, BOX),
    ("b3", "classical", 735, 145, 190, 120, BOX),
    ("o1", "outlabel", 265, 280, 200, 26, "→ ?"),
    ("o2", "outlabel", 500, 280, 200, 26, "→ ?"),
    ("o3", "outlabel", 735, 280, 190, 26, "→ ?"),
    ("dec", "decision", 985, 125, 200, 160, "purity ≥ ?"),
    ("keep", "data", 1240, 50, 210, 100, DATA2),
    ("drop", "data", 1240, 270, 210, 90, "?"),
    ("tr", "learned", 1490, 50, 170, 100, BOX),
    ("ocount", "outlabel", 1225, 160, 250, 26, "how many: ?"),
]
E6 = [
    ("e1", "in", "b1", "", ""), ("e2", "b1", "b2", "", ""), ("e3", "b2", "b3", "", ""),
    ("e4", "b3", "dec", "", ""),
    ("yes", "dec", "keep", ORTHO + "exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;", "yes"),
    ("no", "dec", "drop", ORTHO + "exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;", "no"),
    ("e5", "keep", "tr", "", ""),
]
write("d6_mining_flow", "D6 auto-mined label flow", V6, E6, 1680, 420, (20, 340))
