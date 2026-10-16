"""Replace the two single-box completion groups (g3, g4 + labels gl3, gl4) with one group
around boxes 7-8. All user-entered text is left untouched."""
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
tree = ET.parse(path)
root = tree.find(".//root")
cells = {c.get("id"): c for c in root.findall("mxCell")}

# g3 becomes the merged group: x 190 -> 1030 (covers box 7 at 205-775 and box 8 at 845-1015)
g3 = cells["g3"].find("mxGeometry")
g3.set("width", "840")
# gl3 becomes the merged label
cells["gl3"].set("value", "Completion: partial car → full car")
cells["gl3"].find("mxGeometry").set("width", "360")
for cid in ("g4", "gl4"):
    root.remove(cells[cid])

tree.write(path, encoding="utf-8", xml_declaration=False)
print("merged; remaining ids:", [c.get("id") for c in root.findall("mxCell") if c.get("id", "").startswith(("g", "b"))])
