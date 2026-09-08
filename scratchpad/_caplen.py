import re, glob, statistics
files = glob.glob('docs/writing/thesis/ch*.tex')
lens = []
for f in files:
    s = open(f, encoding='utf-8').read()
    for m in re.finditer(r'\\caption(\[[^\]]*\])?\{', s):
        i = m.end(); depth = 1; j = i
        while j < len(s) and depth:
            if s[j] == '{': depth += 1
            elif s[j] == '}': depth -= 1
            j += 1
        lens.append((len(s[i:j-1]), f.split('/')[-1]))
lens.sort(reverse=True)
print('num captions:', len(lens))
for L, f in lens[:8]:
    print(' ', L, f)
print('median chars:', statistics.median([L for L, _ in lens]))
print('>400 chars:', sum(1 for L, _ in lens if L > 400))
