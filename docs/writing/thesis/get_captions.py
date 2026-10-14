import re

with open(r'D:\Code\LIDAR_Segmentation\docs\writing\thesis\thesis_full.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all \caption[...] or \caption{...} blocks
captions = []
for match in re.finditer(r'\\caption(?:\[.*?\])?\{(.*?)\}', content, re.DOTALL):
    cap = match.group(1).strip()
    # clean up newlines for easy reading
    cap = ' '.join(cap.split())
    captions.append(cap)

for i, cap in enumerate(captions):
    word_count = len(cap.split())
    print(f"[{i+1}] (Words: {word_count}): {cap[:100]}...")
