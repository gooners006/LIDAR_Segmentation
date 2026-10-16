"""D4 scaffold: RANSAC plane fit, side view. Two tries on the same scan slice; the user counts
inliers and fills the decision rule. Scale 40 px = 1 m, so the 0.2 m tolerance = 8 px."""
import math, sys
from xml.sax.saxutils import quoteattr

out = sys.argv[1]
TOL = 8.0  # px  (= 0.2 m)

noise = [-3, 2, -2, -3, 1, -4, 2, 0, -3, 4, -1, 3, 0, 1, -4, 2, 0, -3]
ground = [(30 + 40 * i, 280 + n) for i, n in enumerate(noise)]
car = [(320, 262), (345, 250), (350, 226), (375, 222), (400, 230), (425, 236), (440, 258), (380, 262)]
pole = [(600, 262), (600, 236), (600, 206), (600, 176), (600, 146)]
pts = ground + car + pole

tries = {
    "A": dict(line=((30, 290), (730, 150)), sampled=[(70, 282), (350, 226), (600, 176)]),
    "B": dict(line=((10, 280), (750, 280)), sampled=[(310, 280), (510, 280), (670, 280)]),
}


def dist(p, a, b):
    (x0, y0), (x1, y1), (x2, y2) = p, a, b
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.hypot(y2 - y1, x2 - x1)


key = {}
for name, t in tries.items():
    a, b = t["line"]
    for s in t["sampled"]:
        assert s in pts and dist(s, a, b) < 0.5, (name, s)
    d = [dist(p, a, b) for p in pts]
    near = [x for x in d if abs(x - TOL) < 2.5]
    assert not near, f"borderline points in try {name}: {near}"
    key[name] = sum(x <= TOL for x in d)
print("answer key: inliers", key)

FONT = "fontFamily=Segoe UI;"
cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
n = [0]


def add(value, style, x, y, w, h):
    n[0] += 1
    cells.append(f'<mxCell id="v{n[0]}" value={quoteattr(value)} style={quoteattr(style)} vertex="1" parent="1">'
                 f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry" /></mxCell>')


def line(a, b, style):
    n[0] += 1
    cells.append(f'<mxCell id="e{n[0]}" style={quoteattr(style)} edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{a[0]}" y="{a[1]}" as="sourcePoint" /><mxPoint x="{b[0]}" y="{b[1]}" as="targetPoint" />'
                 f'</mxGeometry></mxCell>')


TXT = "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontSize=19;fontColor=#1F2933;" + FONT
TXTC = TXT.replace("align=left", "align=center")
DOT = "ellipse;html=1;fillColor=#9AA0A6;strokeColor=none;"
RED = "ellipse;html=1;fillColor=#D7263D;strokeColor=#FFFFFF;strokeWidth=2;"
QS = "fontStyle=1;"  # bold for the '?' answer slots

for (name, t), ox in zip(tries.items(), (20, 880)):
    oy = 40
    add("", "rounded=1;arcSize=3;html=1;fillColor=none;strokeColor=#D1D5DB;strokeWidth=2;", ox, oy, 780, 360)
    add(f"<b>Try {name}</b>: plane through three random points", TXT, ox + 16, oy + 6, 600, 30)
    (ax, ay), (bx, by) = t["line"]
    # tolerance band: rectangle centred on the line, rotated to its angle
    L = math.hypot(bx - ax, by - ay); ang = math.degrees(math.atan2(by - ay, bx - ax))
    cx, cy = ox + (ax + bx) / 2, oy + (ay + by) / 2
    add("", f"rounded=0;html=1;fillColor=#D7263D;opacity=15;strokeColor=none;rotation={ang:.2f};",
        cx - L / 2, cy - TOL, L, 2 * TOL)
    for (px, py) in pts:
        add("", DOT, ox + px - 4, oy + py - 4, 8, 8)
    line((ox + ax, oy + ay), (ox + bx, oy + by), "endArrow=none;html=1;strokeColor=#D7263D;strokeWidth=2;dashed=1;")
    for (px, py) in t["sampled"]:
        add("", RED, ox + px - 7, oy + py - 7, 14, 14)
    # fixed object labels
    add("car", TXTC + "fontColor=#6B7280;fontStyle=2;", ox + 330, oy + 186, 100, 26)
    add("pole", TXTC + "fontColor=#6B7280;fontStyle=2;", ox + 560, oy + 112, 80, 26)
    add("road", TXTC + "fontColor=#6B7280;fontStyle=2;", ox + 150, oy + 292, 80, 26)
    # answer slots
    add("inlier band: ± ? m", TXT + "fontColor=#D7263D;", ox + 520, oy + 300, 250, 26)
    add("inliers: ?", TXT + QS + "fontSize=24;", ox + 16, oy + 318, 300, 34)

# scale bar (1 m = 40 px) on the left panel
line((20 + 760, 40 + 140), (20 + 760, 40 + 180), "endArrow=none;startArrow=none;html=1;strokeColor=#1F2933;strokeWidth=3;")
add("1 m", TXT + "fontSize=16;", 20 + 712, 40 + 148, 46, 24)
# decision rule
add("Side view: the ground plane is seen edge-on, so it looks like a line.", TXT + "fontColor=#6B7280;fontStyle=2;fontSize=17;", 20, 6, 900, 28)
add("① Repeat ? times; keep the try with the most inliers → <b>Try ?</b>", TXT, 20, 414, 1640, 30)
add("② Safety check: keep the plane only if its normal points ? (otherwise remove nothing)", TXT, 20, 450, 1640, 30)
add("③ The kept plane's ? inliers are removed as ground; everything else goes on to clustering", TXT, 20, 486, 1640, 30)
# legend
add("", RED, 1380, 14, 14, 14); add("sampled point", TXT + "fontSize=16;", 1400, 8, 140, 26)
add("", DOT, 1543, 17, 8, 8); add("scan point", TXT + "fontSize=16;", 1557, 8, 120, 26)

xml = ('<mxfile host="app.diagrams.net">\n  <diagram name="D4 RANSAC plane fit" id="d4">\n'
       '    <mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
       'page="1" pageScale="1" pageWidth="1680" pageHeight="540" math="0" shadow="0">\n      <root>\n        '
       + "\n        ".join(cells) + "\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n")
open(out, "w", encoding="utf-8").write(xml)
print("cells:", n[0])
