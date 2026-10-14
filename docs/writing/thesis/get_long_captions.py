import re

with open(r'D:\Code\LIDAR_Segmentation\docs\writing\thesis\thesis_full.tex', 'r', encoding='utf-8') as f:
    content = f.read()

captions = []
for match in re.finditer(r'\\caption(?:\[.*?\])?\{(.*?)\}', content, re.DOTALL):
    cap = match.group(1).strip()
    cap = ' '.join(cap.split())
    captions.append(cap)

# Print specific long captions
for idx in [2, 3, 14, 18, 19, 38]: # zero-indexed
    print(f"\n--- Caption {idx+1} ({len(captions[idx].split())} words) ---")
    print(captions[idx])
