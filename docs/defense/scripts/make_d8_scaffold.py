"""D8 scaffold: tracking timeline + track-level filter (slide 2.2.5), for the user to fill in draw.io.

Two lanes, one per track [illustration, not thesis data]:
  top    = a real car the classifier misses in one frame -> whole track accepted, that frame kept
  bottom = a bush the classifier calls 'car' once        -> track rejected, that false positive gone
'?' = values for the user to fill (thesis: fig:tracker, tab:pipeline-config tracker rows).

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/make_d8_scaffold.py docs/defense/assets/d8_tracking_timeline.drawio
"""
import sys
from xml.sax.saxutils import quoteattr

out_path = sys.argv[1]
FONT = "fontFamily=Segoe UI;fontColor=#1F2933;"
ST = {
    "car": "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#DCE8F5;strokeColor=#2B5C8A;strokeWidth=2;fontSize=19;" + FONT,
    "notcar": "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#9AA0A6;strokeWidth=2;dashed=1;fontSize=19;" + FONT,
    "decision": "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#FDE6CC;strokeColor=#D9822B;strokeWidth=2;fontSize=19;" + FONT,
    "accept": "ellipse;whiteSpace=wrap;html=1;fillColor=#D9F2E3;strokeColor=#12A150;strokeWidth=2;fontSize=19;" + FONT,
    "reject": "ellipse;whiteSpace=wrap;html=1;fillColor=#F3F4F6;strokeColor=#3E4C59;strokeWidth=2;fontSize=19;" + FONT,
    "lane": "text;html=1;align=left;verticalAlign=middle;fontSize=22;fontStyle=1;fontFamily=Segoe UI;fontColor=#1F2933;",
    "note": "text;html=1;align=center;verticalAlign=middle;fontSize=18;fontStyle=2;fontFamily=Segoe UI;fontColor=#6B7280;",
    "key": "text;html=1;align=center;verticalAlign=middle;fontSize=19;fontStyle=1;fontFamily=Segoe UI;fontColor=#D9822B;",
    "legendtext": "text;html=1;align=left;verticalAlign=middle;fontSize=18;fontFamily=Segoe UI;fontColor=#3E4C59;",
}
EDGE = ("endArrow=block;endFill=1;html=1;strokeColor=#3E4C59;strokeWidth=3;fontSize=17;fontFamily=Segoe UI;"
        "labelBackgroundColor=#FFFFFF;")
LINK = "endArrow=none;html=1;strokeColor=#2B5C8A;strokeWidth=3;fontSize=17;fontFamily=Segoe UI;labelBackgroundColor=#FFFFFF;"
ORTHO = "edgeStyle=orthogonalEdgeStyle;rounded=1;"


def frame(t, vote):
    return f'<span style="font-size: 17px; color: #6B7280;">frame {t}</span><br><b>{vote}</b>'


X0, DX, W, H = 20, 175, 140, 80
TOP, BOT = 70, 330
car_votes = ["car", "car", "not-car", "car", "car"]          # a car the classifier misses once
bush_votes = ["not-car", "car", "not-car", "not-car"]         # a bush it calls 'car' once

V = [("l1", "lane", 20, TOP - 50, 900, 30, "Track 1: a parked car &nbsp;<span style=\"font-weight:normal;"
      "font-size:19px;color:#6B7280\">the classifier misses it in one frame</span>"),
     ("l2", "lane", 20, BOT - 50, 900, 30, "Track 2: a bush &nbsp;<span style=\"font-weight:normal;"
      "font-size:19px;color:#6B7280\">the classifier calls it a car once</span>")]
E = []
for lane, y, votes in (("a", TOP, car_votes), ("b", BOT, bush_votes)):
    for i, v in enumerate(votes):
        V.append((f"{lane}{i}", "car" if v == "car" else "notcar", X0 + i * DX, y, W, H, frame(i + 1, v)))
        if i:
            E.append((f"{lane}l{i}", f"{lane}{i-1}", f"{lane}{i}", "", "" if i > 1 else "≤ ? m", LINK))

DXD = X0 + 5 * DX + 30                                        # decision column
V += [
    ("d1", "decision", DXD, TOP - 10, 300, 100, "<b>Track vote</b><br>car votes: ? of ?<br>≥ ? votes and ≥ ?% ?"),
    ("d2", "decision", DXD, BOT - 10, 300, 100, "<b>Track vote</b><br>car votes: ? of ?<br>≥ ? votes and ≥ ?% ?"),
    ("r1", "accept", DXD + 360, TOP - 20, 310, 120, "<b>accept the whole track</b><br>frame 3 counted too<br><i>recall ↑</i>"),
    ("r2", "reject", DXD + 360, BOT - 20, 310, 120, "<b>reject the whole track</b><br>frame 2 dropped<br><i>precision ↑</i>"),
    ("gap", "note", X0, TOP + H + 22, 5 * DX - 35, 26, "linked frame to frame by nearest centroid; a track survives ? missed frames"),
    ("why", "key", X0, BOT + H + 26, 5 * DX + 330, 28, "Decide once per object over its whole track, not once per frame"),
    # legend
    ("lc", "car", DXD + 360, BOT + 125, 24, 24, ""), ("lct", "legendtext", DXD + 392, BOT + 125, 220, 24, "classifier: car"),
    ("ln", "notcar", DXD + 360, BOT + 157, 24, 24, ""), ("lnt", "legendtext", DXD + 392, BOT + 157, 220, 24, "classifier: not-car"),
]
E += [("ea", "a4", "d1", "", "", EDGE), ("eb", "b3", "d2", "", "", EDGE),
      ("ea2", "d1", "r1", "", "", EDGE), ("eb2", "d2", "r2", "", "", EDGE)]

cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
for cid, st, x, y, w, h, *val in V:
    v = val[0] if val else ""
    cells.append(f'<mxCell id="{cid}" value={quoteattr(v)} style={quoteattr(ST[st])} vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')
for cid, s, t, extra, label, base in E:
    cells.append(f'<mxCell id="{cid}" value={quoteattr(label)} style={quoteattr(base + extra)} edge="1" '
                 f'parent="1" source="{s}" target="{t}"><mxGeometry relative="1" as="geometry" /></mxCell>')
xml = ('<mxfile host="app.diagrams.net">\n  <diagram name="D8 tracking timeline" id="d8">\n'
       '    <mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
       'page="1" pageScale="1" pageWidth="1700" pageHeight="560" math="0" shadow="0">\n      <root>\n        '
       + "\n        ".join(cells) + "\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n")
open(out_path, "w", encoding="utf-8").write(xml)
print("wrote", out_path, len(V), "vertices", len(E), "edges")
