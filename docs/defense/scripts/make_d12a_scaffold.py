"""D12a scaffold: the plausibility judge, slide 3.4.2, for the user to fill.

Three kinds of cloud (raw scan, completed, dense reference) go through the same preparation into
one independent judge, which says "car" or not; acceptance = the share judged car. '?' = values
for the user to fill (thesis ch4 §Real Data Plausibility, tab:plausibility; answers in S6 notes
Part C). Colours follow the deck: blue = scanned, orange = added by the network, grey = data,
dark = the judge (not part of the pipeline).

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/make_d12a_scaffold.py docs/defense/assets/d12a_judge.drawio
"""
import sys
from xml.sax.saxutils import quoteattr

out_path = sys.argv[1]
FONT = "fontFamily=Segoe UI;fontColor=#1F2933;"
ST = {
    "scan": "ellipse;whiteSpace=wrap;html=1;fillColor=#DCE8F5;strokeColor=#2B5C8A;strokeWidth=2;fontSize=19;" + FONT,
    "comp": "ellipse;whiteSpace=wrap;html=1;fillColor=#FDE6CC;strokeColor=#D9822B;strokeWidth=2;fontSize=19;" + FONT,
    "ref": "ellipse;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#9AA0A6;strokeWidth=2;fontSize=19;" + FONT,
    "prep": "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#3E4C59;strokeWidth=2;fontSize=19;" + FONT,
    "judge": "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#1F2933;strokeWidth=4;fontSize=19;" + FONT,
    "dec": "rhombus;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#3E4C59;strokeWidth=2;fontSize=20;fontStyle=1;" + FONT,
    "out": "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#9AA0A6;strokeWidth=2;fontSize=19;" + FONT,
    "lane": "text;html=1;align=left;verticalAlign=middle;fontSize=22;fontStyle=1;fontFamily=Segoe UI;fontColor=#1F2933;",
    "note": "text;html=1;align=left;verticalAlign=middle;fontSize=18;fontStyle=2;fontFamily=Segoe UI;fontColor=#6B7280;",
}
EDGE = "endArrow=block;endFill=1;html=1;strokeColor=#3E4C59;strokeWidth=3;fontSize=17;fontFamily=Segoe UI;labelBackgroundColor=#FFFFFF;"


def box(title, detail=None):
    t = f'<span style="font-size: 21px;"><b>{title}</b></span>'
    return t + (f"<br>{detail}" if detail else "")


V = [
    ("lane", "lane", 20, 0, 860, 30, "SAME TEST FOR ALL THREE &nbsp;<span style=\"font-weight:normal;"
     "font-size:19px;color:#6B7280\">no ground truth needed</span>"),
    ("raw", "scan", 20, 50, 260, 110, "<b>Raw scan</b><br><i>one frame, N = ?</i>"),
    ("comp", "comp", 320, 50, 260, 110, "<b>Completed</b><br><i>PCN output, same cars</i>"),
    ("ref", "ref", 620, 50, 260, 110, "<b>Dense reference</b><br><i>parked car, many frames, N = ?</i>"),
    ("prep", "prep", 120, 220, 660, 90, box("Same preparation", "turn upright · ? points · unit size")),
    ("judge", "judge", 120, 370, 660, 130, box("Independent judge: PointNet++",
                                              "trained on ? (? object classes),<br>never saw PCN's training shapes")),
    ("dec", "dec", 330, 560, 240, 130, "car?"),
    ("out", "out", 620, 580, 260, 90, box("Acceptance", "share judged “car”")),
    ("n1", "note", 20, 720, 860, 30, "Judge accuracy on its own test set: ? (pass mark fixed in advance: ?)"),
]
E = [
    ("e1", "raw", "prep", "entryX=0.15;entryY=0;entryDx=0;entryDy=0;", ""),
    ("e2", "comp", "prep", "entryX=0.5;entryY=0;entryDx=0;entryDy=0;", ""),
    ("e3", "ref", "prep", "entryX=0.85;entryY=0;entryDx=0;entryDy=0;", ""),
    ("e4", "prep", "judge", "", ""),
    ("e5", "judge", "dec", "", ""),
    ("e6", "dec", "out", "", "yes"),
]

cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
for cid, st, x, y, w, h, v in V:
    cells.append(f'<mxCell id="{cid}" value={quoteattr(v)} style={quoteattr(ST[st])} vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')
for cid, s, t, extra, label in E:
    cells.append(f'<mxCell id="{cid}" value={quoteattr(label)} style={quoteattr(EDGE + extra)} edge="1" '
                 f'parent="1" source="{s}" target="{t}"><mxGeometry relative="1" as="geometry" /></mxCell>')
xml = ('<mxfile host="app.diagrams.net">\n  <diagram name="D12a plausibility judge" id="d12a">\n'
       '    <mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
       'page="1" pageScale="1" pageWidth="900" pageHeight="780" math="0" shadow="0">\n      <root>\n        '
       + "\n        ".join(cells) + "\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n")
open(out_path, "w", encoding="utf-8").write(xml)
print("wrote", out_path, len(V), "vertices", len(E), "edges")
