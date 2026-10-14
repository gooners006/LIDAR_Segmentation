import re

with open(r'D:\Code\LIDAR_Segmentation\docs\writing\thesis\thesis_full.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract lines
lines = content.split('\n')
for i, line in enumerate(lines):
    if '\\begin{equation}' in line:
        start = max(0, i-5)
        end = min(len(lines), i+8)
        print(f"\n--- Context around Equation at line {i+1} ---")
        print("\n".join(lines[start:end]))
