"""D11 scaffold: PCN training (synthetic) vs inference (real), slide 2.3.1, for the user to fill.

Top lane = training on synthetic ShapeNet meshes; bottom lane = inference on a real cluster;
joined by a 'same weights' arrow. '?' = values for the user to fill (thesis ch3 §Completion
Network, §Training Data, §Inference-Time Partial Normalisation; tab:completion-config).
Wording says "shapes"/"mesh", not "car": the training set also holds ShapeNet buses and
motorcycles (S5 notes trap).

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/make_d11_scaffold.py docs/defense/assets/d11_pcn.drawio
"""
import sys
from xml.sax.saxutils import quoteattr

out_path = sys.argv[1]
FONT = "fontFamily=Segoe UI;fontColor=#1F2933;"
ST = {
    "classical": "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#DCE8F5;strokeColor=#2B5C8A;strokeWidth=2;fontSize=19;" + FONT,
    "learned": "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#FDE6CC;strokeColor=#D9822B;strokeWidth=2;fontSize=19;" + FONT,
    "data": "ellipse;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#9AA0A6;strokeWidth=2;fontSize=19;" + FONT,
    "loss": "rhombus;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#3E4C59;strokeWidth=2;fontSize=18;" + FONT,
    "lane": "text;html=1;align=left;verticalAlign=middle;fontSize=22;fontStyle=1;fontFamily=Segoe UI;fontColor=#1F2933;",
    "outlabel": "text;html=1;align=center;verticalAlign=middle;fontSize=18;fontStyle=2;fontFamily=Segoe UI;fontColor=#6B7280;",
    "key": "text;html=1;align=center;verticalAlign=middle;fontSize=20;fontStyle=1;fontFamily=Segoe UI;fontColor=#D9822B;",
    "legendtext": "text;html=1;align=left;verticalAlign=middle;fontSize=18;fontFamily=Segoe UI;fontColor=#3E4C59;",
}
EDGE = "endArrow=block;endFill=1;html=1;strokeColor=#3E4C59;strokeWidth=3;fontSize=17;fontFamily=Segoe UI;labelBackgroundColor=#FFFFFF;"
SAME = ("endArrow=block;endFill=1;html=1;strokeColor=#D9822B;strokeWidth=3;dashed=1;fontSize=19;fontFamily=Segoe UI;"
        "fontStyle=1;fontColor=#D9822B;labelBackgroundColor=#FFFFFF;")
ORTHO = "edgeStyle=orthogonalEdgeStyle;rounded=1;"


def box(title, detail=None):
    t = f'<span style="font-size: 21px;"><b>{title}</b></span>'
    return t + (f"<br>{detail}" if detail else "")


DY = 330
V = [
    ("lane1", "lane", 20, 0, 900, 30, "TRAINING &nbsp;<span style=\"font-weight:normal;font-size:19px;color:#6B7280\">"
     "offline, synthetic shapes only</span>"),
    ("lane2", "lane", 20, DY - 20, 900, 30, "INFERENCE &nbsp;<span style=\"font-weight:normal;font-size:19px;color:#6B7280\">"
     "on a real scan, weights frozen</span>"),
    # ---- training lane
    ("mesh", "data", 20, 80, 190, 100, "ShapeNet mesh<br><i>complete shape</i>"),
    ("gen", "classical", 250, 65, 290, 130, box("Simulated LiDAR view", "one HDL-64E viewpoint, ? – ? m,<br>? m voxels, ground cut")),
    ("pcn1", "learned", 600, 65, 300, 130, box("PCN", "? pts in → ? coarse → ? pts out")),
    ("loss", "loss", 960, 55, 210, 150, "Chamfer<br>loss"),
    ("tgt", "data", 1230, 80, 220, 100, "target:<br><i>full mesh sample</i>"),
    ("o1", "outlabel", 250, 200, 290, 26, "→ partial that looks like a real scan"),
    # ---- inference lane
    ("real", "data", 20, DY + 60, 190, 100, "real car cluster<br><i>one frame</i>"),
    ("gate", "classical", 250, DY + 45, 220, 130, box("L-shape gate", "length ≥ ? m,<br>width ≤ ? m")),
    ("norm", "classical", 510, DY + 45, 270, 130, box("Canonical frame", "rotate · centre (width prior ? m) ·<br>scale ÷ ?")),
    ("pcn2", "learned", 820, DY + 45, 220, 130, box("PCN", "same weights")),
    ("back", "classical", 1080, DY + 45, 200, 130, box("Rotate back", "to the scan")),
    ("out", "data", 1320, DY + 60, 200, 100, "completed car<br><i>? points</i>"),
    ("o2", "outlabel", 510, DY + 180, 270, 26, "→ same frame as in training"),
    ("why", "key", 20, DY + 240, 1500, 30,
     "The bridge: training inputs look like real scans, and real scans are put into the training frame"),
    # legend
    ("lc", "classical", 1250, 250, 24, 24, ""), ("lct", "legendtext", 1282, 250, 230, 24, "classical (training-free)"),
    ("ll", "learned", 1250, 280, 24, 24, ""), ("llt", "legendtext", 1282, 280, 230, 24, "learned"),
]
E = [
    ("e1", "mesh", "gen", "", "", EDGE), ("e2", "gen", "pcn1", "", "", EDGE), ("e3", "pcn1", "loss", "", "", EDGE),
    ("e4", "tgt", "loss", "", "", EDGE), ("e5", "mesh", "tgt",
     ORTHO + "exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;", "sample", EDGE),
    ("f1", "real", "gate", "", "", EDGE), ("f2", "gate", "norm", "", "", EDGE), ("f3", "norm", "pcn2", "", "", EDGE),
    ("f4", "pcn2", "back", "", "", EDGE), ("f5", "back", "out", "", "", EDGE),
    ("same", "pcn1", "pcn2", ORTHO + "exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;",
     "same weights", SAME),
]

cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
for cid, st, x, y, w, h, *val in V:
    v = val[0] if val else ""
    cells.append(f'<mxCell id="{cid}" value={quoteattr(v)} style={quoteattr(ST[st])} vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')
for cid, s, t, extra, label, base in E:
    cells.append(f'<mxCell id="{cid}" value={quoteattr(label)} style={quoteattr(base + extra)} edge="1" '
                 f'parent="1" source="{s}" target="{t}"><mxGeometry relative="1" as="geometry" /></mxCell>')
xml = ('<mxfile host="app.diagrams.net">\n  <diagram name="D11 PCN training and inference" id="d11">\n'
       '    <mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
       'page="1" pageScale="1" pageWidth="1560" pageHeight="640" math="0" shadow="0">\n      <root>\n        '
       + "\n        ".join(cells) + "\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n")
open(out_path, "w", encoding="utf-8").write(xml)
print("wrote", out_path, len(V), "vertices", len(E), "edges")
