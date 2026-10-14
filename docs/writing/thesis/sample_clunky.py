import re
import random

with open(r'D:\Code\LIDAR_Segmentation\docs\writing\thesis\thesis_full.tex', 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(r'%.*?\n', '\n', text)
text = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '', text, flags=re.DOTALL)
text = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', '', text, flags=re.DOTALL)
paragraphs = text.split('\n\n')
long_sentences = []
for p in paragraphs:
    clean_p = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', p)
    sentences = re.split(r'(?<=[.!?]) +', clean_p.replace('\n', ' '))
    for s in sentences:
        if len(s.split()) > 45:
            long_sentences.append(s.strip())

random.seed(42)
for s in random.sample(long_sentences, 10):
    print("- " + s)
