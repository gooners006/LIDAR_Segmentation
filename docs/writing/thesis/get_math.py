import re

with open(r'D:\Code\LIDAR_Segmentation\docs\writing\thesis\thesis_full.tex', 'r', encoding='utf-8') as f:
    content = f.read()

equations = []
for match in re.finditer(r'\\begin\{equation\}(.*?)\\end\{equation\}', content, re.DOTALL):
    equations.append(match.group(1).strip())

for i, eq in enumerate(equations):
    print(f"--- Equation {i+1} ---")
    print(eq)
