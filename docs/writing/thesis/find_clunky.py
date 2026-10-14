import re

with open(r'D:\Code\LIDAR_Segmentation\docs\writing\thesis\thesis_full.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Remove comments
text = re.sub(r'%.*?\n', '\n', text)
# Remove equations and tikz
text = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '', text, flags=re.DOTALL)
text = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', '', text, flags=re.DOTALL)

paragraphs = text.split('\n\n')
long_sentences = []

for p in paragraphs:
    # remove basic latex commands for length checking
    clean_p = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', p)
    sentences = re.split(r'(?<=[.!?]) +', clean_p.replace('\n', ' '))
    for s in sentences:
        words = s.split()
        if len(words) > 45: # extremely long sentence
            long_sentences.append(s.strip())

print(f"Found {len(long_sentences)} sentences with >45 words.")
for i, s in enumerate(long_sentences[:10]):
    print(f"{i+1}: {s}")

