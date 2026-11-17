#!/usr/bin/env bash
# Merge per-section bib files and compile. Usage: bash build.sh
set -e
cd "$(dirname "$0")"
python3 - <<'PY'
import glob,re
seen=set(); out=[]
for f in sorted(glob.glob('bib/*.bib')):
    txt=open(f,encoding='utf-8').read()
    for m in re.finditer(r'@\w+\s*\{\s*([^,\s]+)\s*,.*?\n\}', txt, re.S):
        k=m.group(1).strip()
        if k in seen: continue
        seen.add(k); out.append(m.group(0))
open('refs.bib','w',encoding='utf-8').write('\n\n'.join(out)+'\n')
PY
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { grep -E "^!" -A3 build.log | head -30; exit 1; }
grep -E "Citation .* undefined|Reference .* undefined" build.log | sort -u | head -20 || true
echo "OK: $(pdfinfo main.pdf 2>/dev/null | grep Pages || echo main.pdf built)"
