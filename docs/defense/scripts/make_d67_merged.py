"""Merge the user's filled D6 (label mining) and D7 (classifier) into one draw.io diagram:
top lane = training (D6), bottom lane = inference (D7), joined by a 'trains' arrow.
Box texts are the user's own (copied from their exported .drawio files); only layout changes."""
import sys
from xml.sax.saxutils import quoteattr

out_path = sys.argv[1]
FONT = "fontFamily=Segoe UI;fontColor=#1F2933;"
ST = {
    "classical": "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#DCE8F5;strokeColor=#2B5C8A;strokeWidth=2;fontSize=19;" + FONT,
    "learned": "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#FDE6CC;strokeColor=#D9822B;strokeWidth=2;fontSize=19;" + FONT,
    "data": "ellipse;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#9AA0A6;strokeWidth=2;fontSize=19;" + FONT,
    "decision": "rhombus;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#3E4C59;strokeWidth=2;fontSize=19;" + FONT,
    "clfgroup": "rounded=1;arcSize=4;html=1;fillColor=none;strokeColor=#D9822B;strokeWidth=2;dashed=1;dashPattern=8 6;",
    "lane": "text;html=1;align=left;verticalAlign=middle;fontSize=22;fontStyle=1;fontFamily=Segoe UI;fontColor=#1F2933;",
    "grouplabel": "text;html=1;align=left;verticalAlign=middle;fontSize=20;fontStyle=1;labelBackgroundColor=#FFFFFF;fontFamily=Segoe UI;fontColor=#3E4C59;",
    "clflabel": "text;html=1;align=left;verticalAlign=middle;fontSize=20;fontStyle=1;labelBackgroundColor=#FFFFFF;fontFamily=Segoe UI;fontColor=#D9822B;",
    "outlabel": "text;html=1;align=center;verticalAlign=middle;fontSize=19;fontStyle=2;fontFamily=Segoe UI;fontColor=#6B7280;",
    "keylabel": "text;html=1;align=center;verticalAlign=middle;fontSize=19;fontStyle=1;fontFamily=Segoe UI;fontColor=#D9822B;",
    "legendtext": "text;html=1;align=left;verticalAlign=middle;fontSize=19;fontFamily=Segoe UI;fontColor=#3E4C59;",
}
EDGE = "endArrow=block;endFill=1;html=1;strokeColor=#3E4C59;strokeWidth=3;fontSize=18;fontFamily=Segoe UI;fontStyle=1;labelBackgroundColor=#FFFFFF;"
TRAINS = "endArrow=block;endFill=1;html=1;strokeColor=#D9822B;strokeWidth=3;fontSize=19;fontFamily=Segoe UI;fontStyle=1;fontColor=#D9822B;labelBackgroundColor=#FFFFFF;"
ORTHO = "edgeStyle=orthogonalEdgeStyle;rounded=1;"


def box(title, detail=None):
    t = f'<span style="font-size: 21px;"><b>{title}</b></span>'
    return t + (f"<br>{detail}" if detail else "")


DY = 345  # bottom lane offset (D7 coordinates + DY)
V = [
    # ---- lane headers
    ("lane1", "lane", 20, 0, 700, 30, "TRAINING &nbsp;<span style=\"font-weight:normal;font-size:19px;color:#6B7280\">offline, on labelled SemanticKITTI scans</span>"),
    ("lane2", "lane", 20, 300, 700, 30, "INFERENCE &nbsp;<span style=\"font-weight:normal;font-size:19px;color:#6B7280\">on any new scan, no labels</span>"),
    # ---- top lane: the user's D6
    ("in", "data", 20, 90, 200, 110, "SemanticKITTI scan<br><i>per-point labels</i>"),
    ("b1", "classical", 255, 85, 200, 120, box("Run pipeline", "preprocess →<br>geometric filter")),
    ("b2", "classical", 490, 85, 200, 120, box("Label transfer", "nearest neighbour")),
    ("b3", "classical", 725, 85, 190, 120, box("Majority vote", "per cluster")),
    ("o1", "outlabel", 255, 215, 200, 26, "→ clusters"),
    ("o2", "outlabel", 490, 215, 200, 26, "→ labelled points"),
    ("o3", "outlabel", 725, 215, 190, 26, "→ label + purity"),
    ("dec", "decision", 950, 70, 200, 150, "purity ≥ 75%"),
    ("keep", "data", 1210, 100, 240, 90, "labelled clusters<br><i>car / not-car</i>"),
    ("drop", "data", 1210, 10, 180, 60, "discarded"),
    # ---- bottom lane: the user's D7 (branch boxes kept, the two grey branch frames replaced
    # by one orange frame = the whole classifier, which is what the training lane trains)
    ("clf", "clfgroup", 225, 30 + DY, 1230, 400),
    ("clfl", "clflabel", 240, 18 + DY, 260, 24, "Classifier"),
    ("cl", "data", 20, 165 + DY, 170, 90, "one cluster<br><i>from our pipeline</i>"),
    ("gl1", "grouplabel", 245, 52 + DY, 220, 24, "Point branch"),
    ("gl2", "grouplabel", 245, 267 + DY, 220, 24, "Box branch"),
    ("p1", "classical", 245, 80 + DY, 210, 110, box("Normalise", "512 points, unit sphere")),
    ("p2", "learned", 490, 80 + DY, 230, 110, box("Shared MLP", "3 → 256-D")),
    ("p3", "classical", 755, 80 + DY, 170, 110, box("Max pooling", "per feature")),
    ("f1", "data", 960, 95 + DY, 210, 80, "shape feature<br><i>256-D</i>"),
    ("q1", "classical", 245, 295 + DY, 210, 110, box("OBB features", "extents, volume, aspect ratio, ...")),
    ("q2", "learned", 490, 295 + DY, 230, 110, box("MLP", "8 → 32-D")),
    ("f2", "data", 960, 310 + DY, 210, 80, "size feature<br><i>32-D</i>"),
    ("head", "learned", 1215, 185 + DY, 210, 110, box("Classification head")),
    ("out", "data", 1480, 200 + DY, 200, 80, "car / not-car"),
    ("why", "keylabel", 560, 222 + DY, 460, 28, "order-invariant: shared MLP + max pooling"),
    # ---- one legend for the whole figure
    ("lc", "classical", 1480, 330 + DY, 24, 24, ""), ("lct", "legendtext", 1512, 330 + DY, 230, 24, "classical (training-free)"),
    ("ll", "learned", 1480, 364 + DY, 24, 24, ""), ("llt", "legendtext", 1512, 364 + DY, 230, 24, "learned"),
]
E = [
    ("e1", "in", "b1", "", ""), ("e2", "b1", "b2", "", ""), ("e3", "b2", "b3", "", ""), ("e4", "b3", "dec", "", ""),
    ("yes", "dec", "keep", "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;", "yes"),
    ("no", "dec", "drop", ORTHO + "exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;", "no"),
    ("tr", "keep", "clf", "exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.898;entryY=0;entryDx=0;entryDy=0;", "trains", TRAINS),
    ("c1", "cl", "p1", ORTHO + "exitX=1;exitY=0.25;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;", ""),
    ("c2", "cl", "q1", ORTHO + "exitX=1;exitY=0.75;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;", ""),
    ("c3", "p1", "p2", "", ""), ("c4", "p2", "p3", "", ""), ("c5", "p3", "f1", "", ""),
    ("c6", "q1", "q2", "", ""), ("c7", "q2", "f2", "", ""),
    ("c8", "f1", "head", ORTHO + "entryX=0;entryY=0.25;entryDx=0;entryDy=0;", ""),
    ("c9", "f2", "head", ORTHO + "entryX=0;entryY=0.75;entryDx=0;entryDy=0;", ""),
    ("c10", "head", "out", "", ""),
]

cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
for cid, st, x, y, w, h, *val in V:
    v = val[0] if val else ""
    cells.append(f'<mxCell id="{cid}" value={quoteattr(v)} style={quoteattr(ST[st])} vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')
for cid, s, t, extra, label, *base in E:
    style = (base[0] if base else EDGE) + extra
    cells.append(f'<mxCell id="{cid}" value={quoteattr(label)} style={quoteattr(style)} edge="1" '
                 f'parent="1" source="{s}" target="{t}"><mxGeometry relative="1" as="geometry" /></mxCell>')
xml = ('<mxfile host="app.diagrams.net">\n  <diagram name="D6+D7 classifier training and inference" id="d67">\n'
       '    <mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
       'page="1" pageScale="1" pageWidth="1700" pageHeight="780" math="0" shadow="0">\n      <root>\n        '
       + "\n        ".join(cells) + "\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n")
open(out_path, "w", encoding="utf-8").write(xml)
print("wrote", out_path, len(V), "vertices", len(E), "edges")
